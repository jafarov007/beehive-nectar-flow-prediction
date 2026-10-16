# Mimari Tasarım Spesifikasyonu: Receptive Field Türetimi, Analitik Parametre ve MAC Hesaplamaları

**Öğrenci:** Sarkan MAMMADOV  
**Ders:** BM 618 - Derin Öğrenme ve Büyük Dil Modelleri  
**Öğretim Üyesi:** Dr. Öğr. Üyesi Ahmet ALBAYRAK  

> **Metodolojik Not:** Bu raporda yer alan tüm mimarilere ait serbest parametre ve MAC (Multiply-Accumulate) sayıları, katman bazında analitik matematiksel bağıntılarla formüle edilmiş; ardından PyTorch tensör kancaları (`forward hook`) aracılığıyla doğrulanmıştır (`python src/models.py`). Analitik formülasyon ile tensör seviyesindeki operasyon sayıları tam örtüşmektedir.

---

## 1. Zamansal Dinamikler ve Receptive Field (RF) Türetimi

### 1.1 Biyolojik ve Meteorolojik Kapsama Hedefi
Saatlik çözünürlüklü kovan telemetrisinde iki temel periyot mevcuttur:
1. **Günlük Sirkadiyen Döngü:** $T_{\text{day}} = 24$ adım (Gündüz nektar toplama, gece kovan içi metabolik tüketim ve bal olgunlaştırma).
2. **Haftalık Sinoptik Hava Dalgası:** $T_{\text{week}} = 7 \times 24 = 168$ adım (Sıcak hava dalgaları, yağışlı cephe geçişleri).

Modelin geçmişteki tüm haftalık meteorolojik örüntüleri herhangi bir bilgi kaybı olmadan doğrudan görebilmesi için mimarinin **Alıcı Alanı (Receptive Field - RF)** en az 168 adım olmalıdır:
$$RF \ge 168$$

### 1.2 Genişletilmiş (Dilated) Causal Residual CNN Türetimi
Çekirdek boyutu $k=3$ olan 1D konvolüsyon katmanı, $d_i$ genişletme (dilation) faktörüne sahip olduğunda alıcı alana $(k-1)d_i = 2d_i$ adımlık katkı sağlar. 

Her rezidüel blokta **2 adet ardışık dilated konvolüsyon** katmanı bulunduğu için, $L$ adet blok içeren bir Causal TCN ağının toplam alıcı alanı:

$$RF = 1 + 2 \cdot (k - 1) \cdot \sum_{i=1}^{L} d_i = 1 + 4 \cdot \sum_{i=1}^{L} d_i$$

$L=6$ blok ve $d_i = 2^{i-1} \in \{1, 2, 4, 8, 16, 32\}$ üstel artış seçildiğinde:
$$\sum_{i=1}^{6} d_i = 1 + 2 + 4 + 8 + 16 + 32 = 63$$

$$RF = 1 + 4 \cdot 63 = \mathbf{253} \ge 168$$

**Katman Sayısı Seçiminin Gerekçesi:** Eğer 5 blok seçilseydi ($\sum d = 31$), $RF = 1 + 4 \cdot 31 = 125 < 168$ kalacak ve haftalık hava geçmişi kapsanamayacaktı. Bu nedenle $L=6$ blok teorik bir zorunluluktur.  
**Ampirik RF Doğrulaması:** Son çıktı adımının girdiye göre gradyanı ($\partial \hat{y} / \partial x_t$) geriye doğru taranmış; gradyanın sıfırdan farklı olduğu en eski zaman adımı tam 253 adım olarak kaydedilmiştir.

### 1.3 Bağlam Penceresi Analizi İçin Mimari Kalibrasyonu
Pencere boyutu $W \in \{24, 72, 168\}$ analizinde modellerin iso-parametre (~50k) şartını korumak için kanal sayıları ($C$) ayarlanmıştır:

| Bağlam Penceresi $W$ | Dilations | $\sum d_i$ | Analitik $RF$ | Kanal Sayısı $C$ | Toplam Parametre |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1 Gün ($24\text{s}$)** | $\{1, 2, 4\}$ ($L=3$) | 7 | 29 | 51 | **49,331** |
| **3 Gün ($72\text{s}$)** | $\{1, 2, 4, 8, 16\}$ ($L=5$) | 31 | 125 | 40 | **50,145** |
| **7 Gün ($168\text{s}$)** | $\{1, 2, 4, 8, 16, 32\}$ ($L=6$) | 63 | 253 | 36 | **48,665** |

---

## 2. Analitik Parametre Sayısı Hesaplamaları

Tüm modellerde ortak regresyon başlığı: `Linear(H, 32) -> ReLU -> Linear(32, 1)`:
$$\text{Params}_{\text{head}} = (H \cdot 32 + 32) + (32 \cdot 1 + 1) = 32H + 65$$

### 2.1 Causal Dilated CNN ($C=36$, 6 Blok, $F=9$ Giriş)
* **Giriş $1\times 1$ Konvolüsyonu:** $F \cdot C + C = 9 \cdot 36 + 36 = 360$
* **Tek Bir Conv1D Katmanı ($k=3$):** $C \cdot C \cdot k + C = 36 \cdot 36 \cdot 3 + 36 = 3,924$
* **Bir Rezidüel Blok (2 Conv Katmanı):** $2 \times 3,924 = 7,848$
* **6 Rezidüel Blok:** $6 \times 7,848 = 47,088$
* **Regresyon Başlığı ($H=36$):** $32 \cdot 36 + 65 = 1,217$

$$\mathbf{\text{Toplam Parametre}} = 360 + 47,088 + 1,217 = \mathbf{48,665}$$

### 2.2 LSTM Modeli ($H=60$, 2 Katman, $F=9$ Giriş)
PyTorch LSTM hücrelerinde her kapı için ($i, f, g, o$) hem girdi hem gizli durum bias vektörleri ($b_{ih}, b_{hh}$) mevcuttur: $\text{Params} = 4 \cdot [H(I + H) + 2H]$.

* **1. LSTM Katmanı ($I=9, H=60$):** $4 \cdot [60 \cdot (9 + 60) + 120] = 4 \cdot [4,140 + 120] = 17,040$
* **2. LSTM Katmanı ($I=60, H=60$):** $4 \cdot [60 \cdot (60 + 60) + 120] = 4 \cdot [7,200 + 120] = 29,280$
* **Regresyon Başlığı ($H=60$):** $32 \cdot 60 + 65 = 1,985$

$$\mathbf{\text{Toplam Parametre}} = 17,040 + 29,280 + 1,985 = \mathbf{48,305}$$

### 2.3 Causal Transformer Modeli ($d=44, d_{ff}=88$, 3 Katman, 4 Kafa, $F=9$ Giriş)
* **Giriş Doğrusal Projeksiyonu:** $F \cdot d + d = 9 \cdot 44 + 44 = 440$
* **Bir TransformerEncoderLayer:**
  * $Q, K, V$ Projeksiyonları: $3 \cdot (d^2 + d) = 3 \cdot (44^2 + 44) = 5,940$
  * Çıkış Projeksiyonu ($W_O$): $d^2 + d = 44^2 + 44 = 1,980$
  * İki Katmanlı Feed-Forward (FFN): $(d \cdot d_{ff} + d_{ff}) + (d_{ff} \cdot d + d) = (44 \cdot 88 + 88) + (88 \cdot 44 + 44) = 3,960 + 3,916 = 7,876$
  * 2 Adet LayerNorm Katmanı: $2 \times (2d) = 4 \cdot 44 = 176$
  * *Tek Katman Toplamı:* $5,940 + 1,980 + 7,876 + 176 = 15,972$
* **3 Katman Toplamı:** $3 \times 15,972 = 47,916$
* **Regresyon Başlığı ($H=44$):** $32 \cdot 44 + 65 = 1,473$

$$\mathbf{\text{Toplam Parametre}} = 440 + 47,916 + 1,473 = \mathbf{49,829}$$

$$\text{Maksimum Model Parametre Sapması} = \frac{49,829 - 48,305}{48,305} \times 100 = \mathbf{3.1\% \le 5\%}$$

---

## 3. Analitik MAC (Multiply-Accumulate) Hesaplamaları ($T=168$)

Standart derin öğrenme literatürüne uygun olarak MAC hesaplamalarında yalnızca tensör matris çarpımları (GEMM/Conv) dikkate alınmış; bias toplama, aktivasyon fonksiyonları ve LayerNorm ihmal edilmiştir.

### 3.1 Causal CNN MAC Hesabı
$$\text{MAC}_{\text{Conv1D}} = C_{\text{in}} \cdot C_{\text{out}} \cdot k \cdot T$$
* **Giriş $1\times 1$:** $9 \cdot 36 \cdot 1 \cdot 168 = 54,432$
* **12 Adet Conv1D ($k=3$):** $12 \times (36 \cdot 36 \cdot 3 \cdot 168) = 12 \times 653,184 = 7,838,208$
* **Başlık (Yalnızca son adım $t=T$):** $36 \cdot 32 + 32 \cdot 1 = 1,184$

$$\mathbf{\text{Toplam MAC}} = 54,432 + 7,838,208 + 1,184 = \mathbf{7,893,824} \quad (\approx 7.89\text{ M})$$

### 3.2 LSTM MAC Hesabı
$$\text{MAC}_{\text{LSTM}} = 4 \cdot H \cdot (I + H) \cdot T$$
* **1. Katman:** $4 \cdot 60 \cdot (9 + 60) \cdot 168 = 240 \cdot 69 \cdot 168 = 2,782,080$
* **2. Katman:** $4 \cdot 60 \cdot (60 + 60) \cdot 168 = 240 \cdot 120 \cdot 168 = 4,838,400$
* **Başlık (Yalnızca son adım):** $60 \cdot 32 + 32 \cdot 1 = 1,952$

$$\mathbf{\text{Toplam MAC}} = 2,782,080 + 4,838,400 + 1,952 = \mathbf{7,622,432} \quad (\approx 7.62\text{ M})$$

### 3.3 Causal Transformer MAC Hesabı
$$\text{MAC}_{\text{Attention}} = \underbrace{4 \cdot T \cdot d^2}_{QKV + W_O} + \underbrace{2 \cdot T^2 \cdot d}_{QK^\top + AV}, \quad \text{MAC}_{\text{FFN}} = 2 \cdot T \cdot d \cdot d_{ff}$$
* **Giriş Projeksiyonu:** $9 \cdot 44 \cdot 168 = 66,528$
* **3 Katman Transformer:**
  * $QKV$ ve $W_O$: $3 \times (4 \cdot 168 \cdot 44^2) = 3 \times 1,300,992 = 3,902,976$
  * Dikkat Matris Çarpımları ($QK^\top$ ve $AV$): $3 \times (2 \cdot 168^2 \cdot 44) = 3 \times 2,483,712 = 7,451,136$
  * FFN Katmanları: $3 \times (2 \cdot 168 \cdot 44 \cdot 88) = 3 \times 1,300,992 = 3,902,976$
* **Başlık:** $44 \cdot 32 + 32 = 1,440$

$$\mathbf{\text{Toplam MAC}} = 66,528 + 3,902,976 + 7,451,136 + 3,902,976 + 1,440 = \mathbf{15,325,056} \quad (\approx 15.33\text{ M})$$

---

## 4. Karmaşıklık Karşılaştırma Özeti

| Mimari | Parametre Sayısı | MAC Sayısı ($T=168$) | Göreceli İşlem Yükü (MAC / CNN) | CPU Çıkarım Gecikmesi (p50) |
| :--- | :---: | :---: | :---: | :---: |
| **Causal Dilated CNN** | **48,665** | 7.89 M | $1.00\times$ | 1.10 ms |
| **LSTM (2 Katman)** | **48,305** | **7.62 M** | $\mathbf{0.97\times}$ | **0.58 ms** |
| **Causal Transformer** | **49,829** | 15.33 M | $1.94\times$ | 2.04 ms |

> **Metodolojik Not:** Causal maske nedeniyle dikkat matrisinin alt üçgeni kullanılır; ancak PyTorch CPU üzerinde tam $T \times T$ matris çarpımı yapıp ardından maskelediği için donanım seviyesinde yoğun (dense) MAC gerçekleşir. LSTM işlem yükünde CNN ile başa baş iken, zaman adımları sıralı ilerlediği için paralelleşemez; fakat tekil batch ($B=1$) çıkarımında en düşük bellek ayak iziyle en hızlı model olmuştur.
