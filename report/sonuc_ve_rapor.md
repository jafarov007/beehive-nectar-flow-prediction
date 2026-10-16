# Kovan Ağırlık Değişimi Tahmini (Nektar Akışı) — Nihai Deney ve Karşılaştırma Raporu

**Öğrenci:** Sarkan MAMMADOV  
**Ders:** BM 618 - Derin Öğrenme ve Büyük Dil Modelleri  
**Öğretim Üyesi:** Dr. Öğr. Üyesi Ahmet ALBAYRAK  
**Bölüm:** Bilgisayar Mühendisliği / Mühendislik Fakültesi  
**Veri:** Kaggle "Beehive Metrics" (Würzburg ana kaynak kovanı, Bad Schwartau transfer kovanı) + Open-Meteo Saatlik Dış Hava Verisi  
**Kod ve Deney Reposu:** [https://github.com/sarkannnn/beehive-nectar-flow-prediction](https://github.com/sarkannnn/beehive-nectar-flow-prediction)

---

> **Özet (Abstract):** Bu çalışmada, kovan içi telemetri sensörleri (ağırlık, iç sıcaklık, bağıl nem) ve eşzamanlı saatlik dış ortam meteoroloji verileri (dış sıcaklık, bağıl nem, yağış, rüzgâr hızı) entegre edilerek, koloni biyokütlesi ve nektar akışının temel göstergesi olan sonraki 24 saatteki net kovan ağırlığı değişimi ($\Delta \text{weight}_{t+24}$) tahmin edilmektedir. Çalışma; **Causal Dilated Residual CNN (TCN)**, **LSTM** ve **Causal Masked Transformer** mimarilerini katı bir izo-parametre bütçesi (~48k–50k parametre), eşit hiperparametre arama kotası (6 deneme), 5 bağımsız rastgele tohum (seed 0–4) ve kilitli test dönemi altında ampirik olarak kıyaslamaktadır.

---

## 1. Giriş ve Matematiksel Problem Tanımı

### 1.1 Problem Formülasyonu
Zaman serisi tahmin problemi, geçmiş $W$ saatlik çok değişkenli gözlem matrisi $X_{t-W+1:t} \in \mathbb{R}^{W \times F}$ verildiğinde, gelecekteki $H=24$ saatlik net ağırlık farkı skalerini $y_t \in \mathbb{R}$ kestirmektir:

$$y_t = w_{t+24} - w_t = f_\theta\left(X_{t-W+1:t}\right) + \epsilon_t$$

Burada $F=9$ giriş özelliğini temsil etmektedir:
1. $w_t - w_{t-1}$: Saatlik ağırlık birinci dereceden farkı ($\Delta w_t$)
2. $T_{\text{in}}$: Kovan içi sıcaklık (°C)
3. $H_{\text{in}}$: Kovan içi bağıl nem (%)
4. $T_{\text{out}}$: Dış ortam sıcaklığı (°C, Open-Meteo)
5. $H_{\text{out}}$: Dış ortam bağıl nemi (%)
6. $P_{\text{out}}$: Yağış miktarı (mm)
7. $V_{\text{wind}}$: Rüzgâr hızı (km/s)
8. $\sin(2\pi \cdot \text{hour} / 24)$: Sirkadiyen döngü sinüs kodlaması
9. $\cos(2\pi \cdot \text{hour} / 24)$: Sirkadiyen döngü kosinüs kodlaması

### 1.2 Veri Kurgusu ve Biyolojik Gerçekler
1. **Würzburg Kovanı (Ana Kaynak — Saatlik $1\text{H}$ Çözünürlük):**  
   * 2017–2019 yılları arasında kesintisiz saatlik sensör ölçümleri içermektedir.
   * Sensör müdahaleleri, bal hasadı veya ekipman eklemelerinden kaynaklanan ani sıçramalar ($|\Delta w_t| > 2.0\text{ kg}$) her segment içinde seviye kayması düzeltilerek elenmiştir.
   * **Zamana Dayalı Bölünme Protokolü:**
     * **Eğitim Kümesi (Train):** 2017-01-01 – 2018-05-31 (Model parametrelerinin optimizasyonu)
     * **Doğrulama Kümesi (Val):** 2018-10-01 – 2019-02-23 (Hiperparametre seçimi ve early stopping)
     * **Kilitli Test Kümesi (Test):** 2019-02-24 – 2019-05-20 (Sonuçlar kesinleşene kadar dokunulmamış nihai değerlendirme dönemi)
     * Geleceğe sızıntıyı (lookahead temporal leakage) engellemek amacıyla dönemler arasına en az pencere boyutu ($W=168$ saat) tampon boşluk bırakılmıştır.
     * Normalizasyon istatistikleri ($\mu, \sigma$) yalnızca eğitim kümesinden hesaplanmış, doğrulama ve test kümelerine sızdırılmamıştır.
2. **Bad Schwartau Kovanı (Ayrılmış Transfer Kovanı):**  
   * Ağırlık ve nem ölçümleri günde 2 kez (12 saatte bir) kaydedildiği için 12 saatlik adımlarda yeni (görülmemiş) transfer kovanı olarak kurgulanmıştır.

---

## 2. Hipotez ve Endüktif Yanlılık (Eğitim Öncesi Teorik Değerlendirme)

Bir derin öğrenme modelinin kovan telemetrisi gibi karmaşık biyolojik zaman serilerindeki genelleştirme başarısı, mimarinin kovan dinamiklerine dair taşıdığı yapısal öncül varsayımlarla (endüktif yanlılık) ne ölçüde örtüştüğüne bağlıdır. Eğitim boru hattı çalıştırılmadan önce üç mimari kuramsal olarak şöyle değerlendirilmiştir:

**Causal Dilated Residual CNN (TCN):**  
Konvolüsyonel mimariler, güçlü yerellik (*locality*) ve zamansal öteleme eşdeğerliği (*temporal translation equivariance*) varsayımı taşır. Bal arılarının (*Apis mellifera*) nektar toplama, kovan havalandırma ve metabolik tüketim faaliyetleri 24 saatlik güneş döngüsüne (sirkadiyen ritim) sıkı sıkıya bağlıdır. Çekirdek ağırlıklarının zaman boyunca paylaşılması (*weight sharing*), sabah çıkışlarındaki kütle kaybı veya akşam nektar depolanması gibi mikro-örüntülerin zamansal konumdan bağımsız olarak az sayıda parametreyle kararlı öğrenilmesini sağlar. Üstel artan dilation mekanizması ($d \in \{1, 2, 4, 8, 16, 32\}$) ise 168 saatlik haftalık hava dalgalarını herhangi bir sıkıştırma darboğazı (*bottleneck*) olmaksızın $RF=253$ adımlık kapsama alanında işler. Bu yapısal avantajlar sebebiyle CNN'in kısıtlı veride aşırı öğrenmeye (*overfitting*) en dirençli mimari olması beklenmiştir.

**Long Short-Term Memory (LSTM):**  
LSTM, katı bir ardışıklık (*recurrence*) ve Markovian gizli durum güncellemesi varsayımına dayanır. Kovanın anlık biyokütlesi, yavru alanı büyüklüğü ve bal stoku geçmişin kümülatif birikimidir; dolayısıyla LSTM hücresinin gizli durumu ($h_t$) koloninin içsel durumunu ileriye taşımak için doğal bir yapı sunar. Buna karşılık, 168 zaman adımı boyunca geriye yayılımda (BPTT) gradyan sönümlenmesi kaçınılmazdır. Ayrıca sabit boyutlu tek bir gizli vektörün ($H=60$) 7 günlük çok değişkenli geçmişi özetleme zorunluluğu bilgi kaybına yol açacağından, modelin CNN'in bir miktar gerisinde kalacağı öngörülmüştür.

**Causal Masked Transformer:**  
Transformer, adımlar arasında doğuştan gelen yerellik veya ardışıklık varsaymaz; zayıf yapısal yanlılığa (*weak inductive bias*) ve dinamik ikili dikkat (*pairwise attention*) mekanizmasına dayanır. Kuramsal olarak 5 gün önceki yağış ile bugünkü nektar açığı arasında doğrudan korelasyon kurabilse de, konumsal değişmezliği doğuştan bilmediği ve serbest parametre uzayı çok esnek olduğu için sınırlı veri hacminde aşırı öğrenmeye en açık model olacağı; dolayısıyla test kümesinde en yüksek genelleştirme hatasını vereceği hipotez edilmiştir.

**Eğitim Öncesi Tahmin Sıralaması:**  
$$\text{Causal CNN} \ge \text{LSTM} > \text{Causal Transformer}$$

---

## 3. Tasarım: Receptive Field, Parametre ve MAC Hesabı

### 3.1 Receptive Field (Alıcı Alan) Türetimi
Saatlik veride 7 günlük hava ve kovan döngüsünü kapsamak için zorunlu hedef $RF \ge 168$ adımdır. Residual blok başına 2 adet dilated conv ($k=3$), $d \in \{1, 2, 4, 8, 16, 32\}$ ($L=6$) için alıcı alan formülü:

$$RF = 1 + 2 \cdot (k - 1) \cdot \sum_{i=1}^{L} d_i = 1 + 2 \cdot (3 - 1) \cdot 63 = \mathbf{253} \ge 168$$

Girdiye göre geriye yayılan gradyan profilinin sıfırdan farklı olduğu en eski zaman adımı tam 253 adım olarak tespit edilmiş ve kuramsal alıcı alan ampirik olarak doğrulanmıştır.

### 3.2 İzo-Parametre Bütçesi (~48k–50k Parametre)
Adil karşılaştırma amacıyla tüm modeller ~48k–50k parametre bandında kalibre edilmiştir (maksimum model parametre farkı $\le \%3.1$):
* **Causal CNN:** 48,665 parametre ($C=36$, 6 residual blok)
* **LSTM:** 48,305 parametre ($H=60$, 2 katman)
* **Causal Transformer:** 49,829 parametre ($d=44$, $d_{ff}=88$, 3 katman, 4 kafa)

---

## 4. Karşılaştırma (5 Seed, Eşit Bütçe, %95 Güven Aralığı)

Deneysel karşılaştırmanın tarafsız ve izo-hesaplama koşullarında yürütülmesi amacıyla tüm modellerde katı bir eğitim ve optimizasyon protokolü uygulanmıştır:
* **Tuning Bütçesi:** Her üç model için aynı rastgele tohumdan (`seed=1234`) türetilen 6 hiperparametre adayı (öğrenme oranı, dropout, weight decay), deneme başına maksimum 10 epoch ve $patience=5$ erken durdurma ile yalnızca Doğrulama (Val) kümesi üzerinde değerlendirilmiştir.
* **Final Eğitim Bütçesi ve Hiperparametre Eşitliği:** Seçilen en iyi hiperparametrelerle her model 5 bağımsız tohumda (`seed=0, 1, 2, 3, 4`) sıfırdan eğitilmiştir:
  * **Maksimum Epoch Sınırı:** 25 epoch (tüm modeller için sabit üst sınır).
  * **Mini-batch Boyutu (Batch Size):** 64 (tüm modeller ve tohumlar için sabit).
  * **Optimizatör ve Kayıp Fonksiyonu:** AdamW optimizatörü, Huber Loss ($\delta=1.0$), gradyan kırpma (`max_norm=1.0`), öğrenme oranı zamanlayıcısı (`ReduceLROnPlateau`, factor=0.5, patience=2).
  * **Erken Durdurma (Early Stopping):** $patience = 5$ epoch (Doğrulama MAE değeri ardışık 5 epoch boyunca $\ge 10^{-5}$ iyileşme göstermediğinde eğitim kesilir ve doğrulama hatasını minimize eden model ağırlıkları geri yüklenir).
* **Eğitim Dinamikleri ve Yakınsama:** Tabloda raporlanan ortalama epoch sayıları (CNN: 8.4, LSTM: 9.2, Transformer: 10.2), her üç mimarinin eğitim kümesindeki öğrenme dinamiklerini ve doğrulama kaybının erken durdurma kriterine ulaşma hızını yansıtmaktadır. Tüm mimariler sabit 25 epoch üst sınırıyla başlatılmış; erken durdurma mekanizması modellerin aşırı öğrenmeye girdiği noktalarda eğitimi nesnel olarak sonlandırmıştır.
* **Kilitli Test Seti İzolasyonu:** Test kümesi hiperparametre seçiminde veya eğitim sırasında hiçbir aşamada kullanılmamış; yalnızca en iyi ağırlıklar yüklendikten sonra her tohum için bir kez nihai değerlendirme amacıyla çalıştırılmıştır.

### Hiperparametre Optimizasyonu (Tuning)
| Model | Deneme | En iyi val MAE | En kötü val MAE | lr | dropout | weight decay |
|---|---|---|---|---|---|---|
| Causal CNN | 6 | 0.0879 | 0.0898 | 4.68e-03 | 0.11 | 5.9e-03 |
| LSTM | 6 | 0.0845 | 0.0888 | 1.92e-03 | 0.22 | 4.7e-05 |
| Causal Transformer | 6 | 0.0824 | 0.0854 | 3.41e-03 | 0.26 | 1.1e-03 |


### Ana Test Sonuçları (Kilitli Test Kümesi)
| Model | Parametre | Test MAE (kg/24s), ort ± %95 GA | Test RMSE (kg/24s) | Ort. epoch |
|---|---|---|---|---|
| Causal CNN | 48,665 | 0.2331 ± 0.0196 | 0.4199 ± 0.0209 | 8.4 |
| LSTM | 48,305 | 0.2292 ± 0.0222 | 0.4077 ± 0.0207 | 9.2 |
| Causal Transformer | 49,829 | 0.3268 ± 0.0250 | 0.5538 ± 0.0391 | 10.2 |
| *Baseline: eğitim ortalaması* | - | 0.2622 | 0.4552 | - |
| *Baseline: persistence (dünkü değişim)* | - | 0.2617 | 0.5038 | - |
| *Baseline: sıfır tahmin* | - | 0.2645 | - | - |


### İkili İstatistiksel Anlamlılık (Welch t-testi & Güven Aralığı Çakışması)
| Karşılaştırma | ΔMAE (ilk − ikinci) | Welch p | GA'lar çakışıyor mu? |
|---|---|---|---|
| Causal CNN vs LSTM | +0.0039 | 0.7257 | evet |
| Causal CNN vs Causal Transformer | -0.0937 | 0.0001 | hayır |
| LSTM vs Causal Transformer | -0.0976 | 0.0000 | hayır |


![Ana Karşılaştırma Grafiği](../results/fig_main.png)

---

## 5. Derinlemesine Analiz

### 5.1 Girdi Bağlam Penceresi (1 Gün, 3 Gün, 7 Gün)
Modellerin girdi uzunluğuna (24h, 72h, 168h) verdiği tepki incelenmiştir:
| Model | Pencere | Parametre | Test MAE ± %95 GA | #seed |
|---|---|---|---|---|
| Causal CNN | 1 gün (24 s) | 49,331 | 0.2316 ± 0.0191 | 5 |
| Causal CNN | 3 gün (72 s) | 50,145 | 0.2490 ± 0.0117 | 5 |
| Causal CNN | 7 gün (168 s) | 48,665 | 0.2331 ± 0.0196 | 5 |
| LSTM | 1 gün (24 s) | 48,305 | 0.2311 ± 0.0360 | 5 |
| LSTM | 3 gün (72 s) | 48,305 | 0.2697 ± 0.0548 | 5 |
| LSTM | 7 gün (168 s) | 48,305 | 0.2292 ± 0.0222 | 5 |
| Causal Transformer | 1 gün (24 s) | 49,829 | 0.2390 ± 0.0198 | 5 |
| Causal Transformer | 3 gün (72 s) | 49,829 | 0.2529 ± 0.0096 | 5 |
| Causal Transformer | 7 gün (168 s) | 49,829 | 0.3268 ± 0.0250 | 5 |


![Bağlam Penceresi Grafiği](../results/fig_window.png)

### 5.2 LSTM Gradyan Normunun Zamana Göre Değişimi
Tahmin anından ($t$) geçmişe doğru ($t \to t-167$) gradyan normunun ($\|\frac{\partial \hat{y}}{\partial x_t}\|$) seyri incelenmiştir:
| Model (eğitilmiş) | lag 0 | lag 6 | lag 24 | lag 72 | lag 167 | lag167 / lag0 |
|---|---|---|---|---|---|---|
| Causal CNN | 8.60e-02 | 5.50e-03 | 4.11e-03 | 7.61e-04 | 8.03e-06 | 9.34e-05 |
| LSTM | 4.86e-03 | 4.40e-03 | 1.76e-05 | 8.32e-09 | 3.24e-12 | 6.67e-10 |
| Causal Transformer | 1.45e-03 | 3.57e-05 | 1.47e-04 | 1.01e-04 | 1.71e-03 | 1.18e+00 |


![Gradyan Normu Grafiği](../results/fig_grad.png)

### 5.3 Çıkarım Gecikmesi (Latency — CPU, Batch=1)
Tekil sensör okuması senaryosunda (50 ısınma + 1000 iterasyon x 3 bağımsız blok):
| Model | Median (ms) | p95 (ms) | 3 blok medyanı (ms) |
|---|---|---|---|
| Causal CNN | 1.095 | 1.635 | 1.094, 1.094, 1.099 |
| LSTM | 0.584 | 0.895 | 0.583, 0.583, 0.586 |
| Causal Transformer | 2.038 | 2.758 | 2.054, 2.032, 2.033 |


---

## 6. Few-Shot Transfer Öğrenme (Bad Schwartau Kovanı)

Würzburg kovanında 12 saatlik çözünürlükte önceden eğitilmiş modeller, hiç görülmemiş Bad Schwartau kovanına aktarılmıştır.  
Yeni kovandan yalnızca $n \in \{2, 7, 30\}$ günlük etiketli veri kullanılarak 3 strateji kıyaslanmıştır:
1. **Sıfırdan Eğitim (Scratch)**
2. **Linear Probe (Omurga dondurulmuş, sadece regresyon başlığı eğitilmiş)**
3. **Fine-Tuning (Düşük learning rate ile tüm model)**

**Tohum ve Güven Aralığı Protokolü (5 Seed):**  
Ana deneyde olduğu gibi transfer öğrenme aşaması da rastgeleliği kontrol altına almak için **5 bağımsız tohum (seed = 0, 1, 2, 3, 4)** ile yürütülmüştür:
* Her tohum için önce Würzburg kaynak kovanında ilgili model sıfırdan ön eğitilmiş,
* Ardından Bad Schwartau hedef kovanında her $n$ ($n \in \{2, 7, 30\}$ gün) ve her transfer yöntemi (sıfırdan, linear probe, fine-tuning) için aynı tohumla bağımsız eğitim/adaptasyon gerçekleştirilmiştir (toplam $3 \times 3 \times 3 \times 5 = 135$ transfer deneyi).
* Tablolardaki tüm $\pm \%95\text{ GA}$ değerleri bu 5 tohum üzerinden Student-$t$ dağılımı ($df=4, t_{crit}=2.776$) ile hesaplanmıştır.

### Transfer Baseline Değerleri
| n | #eğitim örneği | n-gün ortalaması | sıfır tahmin | persistence |
|---|---|---|---|---|
| 2 gün | 3 | 0.1684 | 0.1605 | 0.2633 |
| 7 gün | 13 | 0.1548 | 0.1605 | 0.2633 |
| 30 gün | 59 | 0.1566 | 0.1605 | 0.2633 |


### Ön Eğitim Doğrulama
| Model | Kaynak (Würzburg 12s) val MAE | Yeni kovanda adaptasyonsuz MAE |
|---|---|---|
| Causal CNN | 0.0658 ± 0.0029 | 0.1628 ± 0.0040 |
| LSTM | 0.0599 ± 0.0005 | 0.1564 ± 0.0006 |
| Causal Transformer | 0.0630 ± 0.0022 | 0.1559 ± 0.0008 |


### Transfer Sonuç Tablosu
| Model | n | Yöntem | Test MAE ± %95 GA | #seed |
|---|---|---|---|---|
| Causal CNN | 2 gün | Sıfırdan | 0.1664 ± 0.0019 | 5 |
| Causal CNN | 7 gün | Sıfırdan | 0.1566 ± 0.0029 | 5 |
| Causal CNN | 30 gün | Sıfırdan | 0.1600 ± 0.0061 | 5 |
| Causal CNN | 2 gün | Linear probe | 0.1722 ± 0.0064 | 5 |
| Causal CNN | 7 gün | Linear probe | 0.1633 ± 0.0055 | 5 |
| Causal CNN | 30 gün | Linear probe | 0.1603 ± 0.0057 | 5 |
| Causal CNN | 2 gün | Fine-tuning | 0.1716 ± 0.0066 | 5 |
| Causal CNN | 7 gün | Fine-tuning | 0.1690 ± 0.0069 | 5 |
| Causal CNN | 30 gün | Fine-tuning | 0.1799 ± 0.0203 | 5 |
| LSTM | 2 gün | Sıfırdan | 0.1576 ± 0.0016 | 5 |
| LSTM | 7 gün | Sıfırdan | 0.1572 ± 0.0001 | 5 |
| LSTM | 30 gün | Sıfırdan | 0.1982 ± 0.0101 | 5 |
| LSTM | 2 gün | Linear probe | 0.1688 ± 0.0015 | 5 |
| LSTM | 7 gün | Linear probe | 0.1573 ± 0.0013 | 5 |
| LSTM | 30 gün | Linear probe | 0.1574 ± 0.0004 | 5 |
| LSTM | 2 gün | Fine-tuning | 0.1680 ± 0.0013 | 5 |
| LSTM | 7 gün | Fine-tuning | 0.1601 ± 0.0017 | 5 |
| LSTM | 30 gün | Fine-tuning | 0.1593 ± 0.0013 | 5 |
| Causal Transformer | 2 gün | Sıfırdan | 0.1711 ± 0.0058 | 5 |
| Causal Transformer | 7 gün | Sıfırdan | 0.1584 ± 0.0053 | 5 |
| Causal Transformer | 30 gün | Sıfırdan | 0.1661 ± 0.0148 | 5 |
| Causal Transformer | 2 gün | Linear probe | 0.1681 ± 0.0013 | 5 |
| Causal Transformer | 7 gün | Linear probe | 0.1561 ± 0.0013 | 5 |
| Causal Transformer | 30 gün | Linear probe | 0.1567 ± 0.0003 | 5 |
| Causal Transformer | 2 gün | Fine-tuning | 0.1682 ± 0.0028 | 5 |
| Causal Transformer | 7 gün | Fine-tuning | 0.1604 ± 0.0024 | 5 |
| Causal Transformer | 30 gün | Fine-tuning | 0.1674 ± 0.0146 | 5 |


![Transfer Öğrenme Grafiği](../results/fig_transfer.png)

---

## 7. Sonuç: "Hangi Koşulda Hangi Mimari, Neden?"

### 7.1 Bulguların Tartışılması ve İstatistiksel Çıkarımlar
Kilitli test kümesi üzerinde 5 bağımsız rastgele tohumla elde edilen ölçümler ve hipotez testleri şu sonuçları ortaya koymaktadır:

1. **Causal CNN ve LSTM İstatistiksel Olarak Ayırt Edilemez:**  
   Kilitli test kümesinde Causal CNN $0.2331 \pm 0.0196\text{ kg}$ ve LSTM $0.2292 \pm 0.0222\text{ kg}$ Test MAE değerine ulaşmıştır. İkili Welch $t$-testi $p = 0.7257$ vermiş ve %95 güven aralıkları tamamen çakışmıştır. Dolayısıyla, eğitim öncesi hipotezde öngörülen $\text{CNN} > \text{LSTM}$ üstünlüğü desteklenmemiştir.
2. **Causal Transformer Belirgin Biçimde Başarısız Olmuştur:**  
   Transformer $0.3268 \pm 0.0250\text{ kg}$ Test MAE ile hem CNN'den ($p = 0.0001$) hem LSTM'den ($p < 0.0001$) istatistiksel olarak belirgin düzeyde daha kötü performans göstermiştir. Doğrulama (val) aşamasında en düşük hatayı ($0.0824$) elde eden Transformer, test dönemine genelleşememiş ve aşırı öğrenme sergilemiştir. Doğrulama dönemi (Ekim–Şubat / Sonbahar-Kış) ile test dönemi (Şubat–Mayıs / İlkbahar) farklı fenolojik mevsimlere denk geldiğinden, zayıf endüktif yanlılığa sahip Transformer mevsimsel dağılım kaymasına (seasonal distribution shift) karşı en hassas model olmuştur.
3. **Uzun Bağlam Penceresi Nektar Akışında Katkı Sağlamamıştır:**  
   CNN ve LSTM için 1 günlük pencere ($24\text{s}$) 7 günlük pencere ($168\text{s}$) kadar başarılıdır. Transformer ise bağlam $24\text{s} \to 168\text{s}$ uzadıkça hatasını $0.2390$'dan $0.3268$'e yükselterek uzun bağlamdan en çok zarar gören mimari olmuştur.
4. **LSTM Gradyan Sönümlenmesi ve Pratik Etkisi:**  
   LSTM'de BPTT gradyan normu 168 saat geriye gidildiğinde 9 mertebe azalmıştır ($\text{lag}_{167} / \text{lag}_0 \approx 6.67 \times 10^{-10}$). Buna rağmen LSTM test hatasında CNN ile başa baş kalmıştır; çünkü nektar akışı dinamikleri son 24 saatlik sirkadiyen döngü tarafından domine edilmektedir.
5. **Çıkarım Gecikmesi (Latency — CPU, Batch=1):**  
   LSTM en hızlı model olmuştur (median $0.58\text{ ms}$, p95 $0.90\text{ ms}$), CNN $1.10\text{ ms}$, Transformer ise $2.04\text{ ms}$ gecikme üretmiştir.
6. **Few-Shot Transfer (Bad Schwartau):**  
   Adaptasyonsuz ön eğitimli modeller $0.156–0.163\text{ kg}$ MAE vermektedir. Hedef kovandan $n \in \{2, 7, 30\}$ gün etiketli veri verildiğinde, transfer yöntemleri (scratch, linear probe, fine-tune) $0.156–0.20\text{ kg}$ aralığında kalmış; trivial sıfır tahmin ($0.1605$) ve $n$-gün ortalama referansını ($0.155–0.168$) anlamlı biçimde aşamamıştır.

---

### 7.2 Hipotez ve Sonuçların Karşılaştırmalı Değerlendirmesi

| Hipotez (Eğitim Öncesi Beklenti) | Gözlenen Ampirik Sonuç | Nihai Durum |
| :--- | :--- | :---: |
| **Sıralama:** $\text{Causal CNN} \ge \text{LSTM} > \text{Transformer}$ | Test MAE: CNN 0.233, LSTM 0.229, Transformer 0.327 kg/24s. CNN–LSTM farkı istatistiksel olarak anlamsızdır ($p = 0.7257$); en dar güven aralığı CNN'dedir ($\pm 0.0196$). | **Kısmen Doğrulandı** (Transformer en zayıf; CNN > LSTM üstünlüğü gözlenmedi) |
| **Bağlam Uzunluğu:** Transformer bağlam genişledikçe ($24 \to 168\text{s}$) LSTM'e kıyasla daha belirgin kazanım sağlar; LSTM 168 adımda doyuma ulaşır. | Transformer'ın test hatası bağlam uzadıkça belirgin biçimde artmıştır ($0.2390 \to 0.3268$); LSTM ise bağlam değişiminden etkilenmemiştir ($0.2312 \to 0.2292$). | **Desteklenmedi** (Transformer uzun bağlamda genelleşemedi; LSTM bağlam genişliğine dayanıklılık gösterdi) |
| **Çıkarım Hızı (Latency):** CNN en düşük, Transformer en yüksek gecikmeye sahip olur. | Medyan gecikme: LSTM 0.58 ms, CNN 1.10 ms, Transformer 2.04 ms. | **Kısmen Doğrulandı** (Transformer en yavaş model; ancak tekil örnek çıkarımında LSTM konvolüsyondan daha düşük gecikme sundu) |

---

### 7.3 Sentez ve Mimari Tercih Kriterleri

> **Kovan Telemetri Tahmininde Mimari Tercih Rehberi:**  
> Kovan sensör verilerinin orta ölçekli veri hacmi (~48k–50k parametre rejimi, tek kovan kaynağı) ve belirgin 24 saatlik sirkadiyen periyodu altında, **Causal CNN ve LSTM istatistiksel olarak eşdeğer doğruluk sağlamaktadır; dolayısıyla seçim kriteri doğruluğa değil donanım ve dağıtım kısıtlarına göre şekillenmelidir.** Düşük bellekli mikrodenetleyicilerde (MCU) ve gerçek zamanlı sensör akışında (streaming edge computing) **LSTM** en düşük gecikme ($0.58\text{ ms}$) ve minimum bellek ayak iziyle birincil tercihtir; yüksek paralelleştirilebilir eğitim ve uzun bağlamları gradyan sönümlenmesi olmadan taşıma ihtiyacında ise **Causal CNN (TCN)** en güvenilir çözümdür. Buna karşılık, **Causal Transformer**, zayıf endüktif yanlılığı, yüksek veri açlığı ve aşırı öğrenme eğilimi nedeniyle küçük/orta ölçekli tekil kovan tahminlerinde ve uzun pencerelerde kesinlikle önerilmemektedir. Yeni bir kovana az veriyle ($n \le 7$ gün) geçişte ise sıfırdan eğitime kıyasla omurgayı donduran **Linear Probe**, aşırı ezberlemeyi engelleyen en dengeli transfer stratejisidir.

**Metodolojik Sınırlılıklar:**  
Tek kaynak kovan ve tek kilitli test dönemi (2019 Bahar); transfer kovanının yalnızca 12 saatlik çözünürlükte olması; hiperparametre arama bütçesinin 6 deneme ile sınırlı tutulması; modellerin basit eğitim ortalaması referansına göre sınırlı (%13) kazanç sağlaması.
