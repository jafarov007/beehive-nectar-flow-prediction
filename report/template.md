# Kovan Ağırlık Değişimi Tahmini (Nektar Akışı) — Nihai Deney ve Karşılaştırma Raporu

**Öğrenci:** Serxan M.  
**Ders:** Derin Öğrenme / Zaman Serileri Analizi  
**Veri:** Kaggle "Beehive Metrics" (Würzburg ana kaynak kovanı, Bad Schwartau transfer kovanı) + Open-Meteo Saatlik Dış Hava Verisi  
**Kod ve Deney Reposu:** `C:\Users\serxan.m\Desktop\kovan_project`

---

## 1. Giriş ve Veri Kurgusu

Bu çalışmada, kovan içi sensörlerinden (ağırlık, iç sıcaklık, iç nem) ve dış ortam meteoroloji verilerinden (dış sıcaklık, bağıl nem, yağış, rüzgâr hızı) yararlanılarak sonraki 24 saatteki kovan ağırlığı değişimi ($\Delta \text{weight}_{t+24}$, nektar akışı) tahmin edilmektedir.

### Veri Seti ve Biyolojik Gerçekler
1. **Würzburg Kovanı (Ana Kaynak):**  
   * 2017–2019 yılları arasında kesintisiz saatlik ($1\text{H}$) sensör ölçümleri içermektedir.
   * Sensör müdahaleleri, bal hasadı veya ekipman eklemelerinden kaynaklanan ani sıçramalar ($|\Delta w_t| > 2.0\text{ kg}$) her segment içinde seviye kayması düzeltilerek elenmiştir.
   * **Bölünme Stratejisi (Zamana Dayalı):**
     * **Eğitim:** 2017-01-01 – 2018-05-31
     * **Doğrulama (Val):** 2018-10-01 – 2019-02-23 (Hiperparametre seçimi için)
     * **Test (Kilitli):** 2019-02-24 – 2019-05-20 (Sonuçlar kesinleşene kadar dokunulmamış kilitli dönem)
     * Sızıntıyı (lookahead leakage) engellemek amacıyla dönemler arasına en az pencere boyutu (168 saat) tampon boşluk bırakılmıştır.
     * Normalizasyon istatistikleri ($\mu, \sigma$) **yalnızca eğitim kümesinden** hesaplanmıştır.
2. **Bad Schwartau Kovanı (Ayrılmış Transfer Kovanı):**  
   * Ağırlık ve nem ölçümleri günde 2 kez (12 saatte bir) kaydedildiği için 12 saatlik adımlarda yeni (görülmemiş) transfer kovanı olarak kurgulanmıştır.

---

## 2. Hipotez (Eğitimden Önce — Endüktif Yanlılık Çerçevesi)

Model performanslarını belirleyen en temel teorik unsur, seçilen mimarinin kovan biyolojisine ve zaman serisi dinamiğine dair taşıdığı öncül yapısal varsayımlardır:

1. **Causal Dilated Residual CNN (TCN):**  
   * *Endüktif Yanlılık:* Güçlü yerellik (locality) ve zamansal öteleme eşdeğerliği (temporal translation equivariance).  
   * *Beklenti:* 24 saatlik döngüsel biyolojik ritim her gün kendini tekrarlar. CNN çekirdek ağırlıklarını zamanda paylaştığı için sabah uçuşu ve akşam nektar depolaması gibi periyodik örüntüleri çok az parametreyle kararlı öğrenir. Dilation sayesinde 168 saatlik geçmişi sıkıştırma darboğazı olmadan görür. Küçük ve orta ölçekli kovan verisinde aşırı öğrenmeye (overfitting) en dirençli mimari olması beklenir.
2. **LSTM (Long Short-Term Memory):**  
   * *Endüktif Yanlılık:* Ardışıklık (recurrence) ve Markovian gizli durum sıkıştırması.  
   * *Beklenti:* Kovanın anlık biyokütlesi birikimli bir durumdur; LSTM hücresi bunu taşımada doğaldır. Ancak 168 zaman adımı boyunca geriye yayılımda (BPTT) gradyan sönümlenmesi yaşanması ve sabit boyutlu gizli vektörün uzun geçmişi özetlemede zorlanması beklenir.
3. **Causal Masked Transformer:**  
   * *Endüktif Yanlılık:* Global dinamik dikkat ve zayıf yapısal yanlılık (weak inductive bias).  
   * *Beklenti:* Tüm adımlar arası doğrudan korelasyon kurabilir; ancak konumsal değişmezliği doğuştan bilmediği ve yüksek veri hacmi istediği için bu veri ölçeğinde aşırı öğrenmeye en açık model olacaktır.

*Eğitim Öncesi Tahmin Sıralaması:*  
$$\text{Causal CNN} \ge \text{LSTM} > \text{Causal Transformer}$$

---

## 3. Tasarım: Receptive Field, Parametre ve MAC Hesabı

* **Receptive Field (RF) Türetimi:**  
  Saatlik veride 7 günlük hava ve kovan döngüsünü kapsamak için zorunlu hedef: $RF \ge 168$ adım.  
  Residual blok başına 2 adet dilated conv ($k=3$), $d \in \{1, 2, 4, 8, 16, 32\}$ için:
  $$RF = 1 + 2 \cdot (k - 1) \cdot \sum_{i=1}^{L} d_i = 1 + 2 \cdot (3 - 1) \cdot 63 = 253 \ge 168 \;\checkmark$$
  *(Ampirik gradyan testi ile doğrulanan RF: tam 253 adım).*
* **İzo-Parametre Bütçesi (~48k–50k Parametre):**
  * Causal CNN: 48,665 parametre ($C=36$, 6 blok)
  * LSTM: 48,305 parametre ($H=60$, 2 katman)
  * Causal Transformer: 49,829 parametre ($d=44$, $d_{ff}=88$, 3 katman, 4 kafa)

---

## 4. Karşılaştırma (5 Seed, Eşit Bütçe, %95 Güven Aralığı)

Ödev şartı gereği karşılaştırmanın tam anlamıyla adil ve izo-koşullarda olması için tüm modellerde **aynı eğitim ve tuning bütçesi protokolü** katı şekilde uygulanmıştır:
* **Tuning Bütçesi:** Her üç model için de aynı rastgele tohumdan (`seed=1234`) türetilen 6 hiperparametre adayı (learning rate, dropout, weight decay), deneme başına maksimum 10 epoch ve $patience=5$ erken durdurma ile yalnızca Doğrulama (Val) kümesi üzerinde değerlendirilmiştir.
* **Final Eğitim Bütçesi ve Hiperparametre Eşitliği:** Seçilen en iyi hiperparametrelerle her model 5 bağımsız tohumda (`seed=0, 1, 2, 3, 4`) sıfırdan eğitilmiştir. Her üç model için:
  * **Maksimum Epoch Sınırı:** 25 epoch (tüm modeller için sabit üst sınır).
  * **Mini-batch Boyutu (Batch Size):** 64 (tüm modeller ve tohumlar için sabit).
  * **Optimizatör ve Kayıp Fonksiyonu:** AdamW optimizatörü, Huber Loss ($\delta=1.0$), gradyan kırpma (gradient clipping, `max_norm=1.0`), öğrenme oranı zamanlayıcısı (`ReduceLROnPlateau`, factor=0.5, patience=2).
  * **Erken Durdurma (Early Stopping):** $patience = 5$ epoch (Doğrulama MAE değeri ardışık 5 epoch boyunca $\ge 10^{-5}$ iyileşme göstermediğinde eğitim otomatik kesilir ve hafızadaki en iyi doğrulama kontrol noktası [checkpoint] geri yüklenir).
* **Ortalama Epoch Farkının Sebebi (8.4 / 9.2 / 10.2):** Tablodaki ortalama epoch sayılarının küçük farklılıklar göstermesi (CNN: 8.4, LSTM: 9.2, Transformer: 10.2), modellere farklı eğitim süresi verilmesinden **değildir**. Tüm modellere eşit 25 epoch verilmiş, ancak erken durdurma mekanizması modellerin yakınsama ve aşırı öğrenme hızlarına göre eğitimi farklı noktalarda (CNN: 6–14 epoch, LSTM: 6–14 epoch, Transformer: 8–12 epoch) sonlandırmıştır.
* **Kilitli Test Seti İzolasyonu:** Test kümesi hiperparametre seçiminde, epoch belirlemede veya erken durdurmada **asla kullanılmamış**; sadece en iyi ağırlıklar yüklendikten sonra her tohum için yalnızca bir kez nihai metrik ölçümü yapılmıştır.

### Hiperparametre Optimizasyonu (Tuning)
{{TABLE_TUNING}}

### Ana Test Sonuçları (Kilitli Test Kümesi)
{{TABLE_MAIN}}

### İkili İstatistiksel Anlamlılık (Welch t-testi & Güven Aralığı Çakışması)
{{TABLE_PAIRS}}

![Ana Karşılaştırma Grafiği](../results/fig_main.png)

---

## 5. Derinlemesine Analiz

### 5.1 Girdi Bağlam Penceresi (1 Gün, 3 Gün, 7 Gün)
Modellerin girdi uzunluğuna (24h, 72h, 168h) verdiği tepki incelenmiştir:
{{TABLE_WINDOW}}

![Bağlam Penceresi Grafiği](../results/fig_window.png)

### 5.2 LSTM Gradyan Normunun Zamana Göre Değişimi
Tahmin anından ($t$) geçmişe doğru ($t \to t-167$) gradyan normunun ($\|\frac{\partial \hat{y}}{\partial x_t}\|$) seyri incelenmiştir:
{{TABLE_GRAD}}

![Gradyan Normu Grafiği](../results/fig_grad.png)

### 5.3 Çıkarım Gecikmesi (Latency — CPU, Batch=1)
Tekil sensör okuması senaryosunda (50 ısınma + 1000 iterasyon x 3 bağımsız blok):
{{TABLE_LATENCY}}

---

## 6. Few-Shot Transfer Öğrenme (Bad Schwartau Kovanı)

Würzburg kovanında 12 saatlik çözünürlükte önceden eğitilmiş modeller, hiç görülmemiş Bad Schwartau kovanına aktarılmıştır.  
Yeni kovandan yalnızca $n \in \{2, 7, 30\}$ günlük etiketli veri kullanılarak 3 strateji kıyaslanmıştır:
1. **Sıfırdan Eğitim (Scratch)**
2. **Linear Probe (Omurga dondurulmuş, sadece regresyon başlığı eğitilmiş)**
3. **Fine-Tuning (Düşük learning rate ile tüm model)**

**Tohum ve Güven Aralığı Protokolü (5 Seed):**  
Ana deneyde olduğu gibi, transfer öğrenme aşaması da rastgeleliği kontrol altına almak için **5 bağımsız tohum (seed = 0, 1, 2, 3, 4)** ile yürütülmüştür:
* Her tohum için önce Würzburg kaynak kovanında ilgili model sıfırdan ön eğitilmiş,
* Ardından Bad Schwartau hedef kovanında her $n$ ($n \in \{2, 7, 30\}$ gün) ve her transfer yöntemi (sıfırdan, linear probe, fine-tuning) için aynı tohumla bağımsız eğitim/adaptasyon gerçekleştirilmiştir (toplam $3 \times 3 \times 3 \times 5 = 135$ transfer deneyi).
* Tablolardaki tüm $\pm \%95\text{ GA}$ değerleri bu 5 tohum üzerinden Student-$t$ dağılımı ($df=4, t_{crit}=2.776$) ile hesaplanmıştır; tabloda her satır için `#seed = 5` olarak belirtilmiştir.

### Transfer Baseline Değerleri
{{TABLE_TRANSFER_BASE}}

### Ön Eğitim Doğrulama
{{TABLE_TRANSFER_PRE}}

### Transfer Sonuç Tablosu
{{TABLE_TRANSFER}}

![Transfer Öğrenme Grafiği](../results/fig_transfer.png)

---

## 7. Sonuç: "Hangi Koşulda Hangi Mimari, Neden?"

Aşağıdaki sonuçlar yalnızca bu çalışmadaki ölçümlerden (tek kaynak kovan, tek test dönemi, 5 seed) çıkarılmıştır.

**Ana bulgular**

1. **Causal CNN ve LSTM istatistiksel olarak ayırt edilemez.** Test MAE 0.233 ve 0.229 kg/24s, Welch p = 0.73, %95 güven aralıkları çakışıyor. Hipotezdeki "CNN ≥ LSTM" sıralamasının CNN > LSTM kısmı **doğrulanmadı**.
2. **Causal Transformer belirgin biçimde daha kötü** (0.327 kg/24s; CNN ve LSTM'e göre p ≤ 0.0001). Hipotezdeki "Transformer en zayıf" tahmini doğrulandı. Transformer validation MAE'de en iyi (0.0824) olduğu halde testte en kötü olmuştur: validation dönemine uyum sağlayıp test dönemine genelleşememiştir (zayıf endüktif yanlılıkla tutarlı, ancak bu çalışma nedensel mekanizmayı ayrıca kanıtlamamaktadır).
3. **Uzun bağlam bu veride yarar sağlamadı.** CNN ve LSTM için 1 günlük pencere 7 günlük pencere kadar iyidir (güven aralıkları çakışıyor). Transformer 24 saatlik pencerede 0.239'a inerken 168 saatte 0.327'ye çıkmaktadır; yani en çok uzun bağlamdan zarar gören modeldir. Haftalık hava örüntüsünün bu veriyle öğrenilebildiğine dair kanıt bulunmamıştır.
4. **Modeller basit referanslara göre yalnızca sınırlı üstündür.** En iyi model (LSTM, 0.229) eğitim ortalaması referansına (0.262) göre yaklaşık %13 daha iyidir. Persistence referansı MAE'de (0.262) ortalama referansıyla aynıdır, RMSE'de ise (0.504) modellerden (0.41–0.42) belirgin kötüdür. Nektar akışı tahmini bu veriyle zordur ve modeller çoğunlukla ortalama davranışı yakalamaktadır.
5. **LSTM'de gradyan sönümlenmesi gözlendi:** lag 167 / lag 0 gradyan oranı ≈ 7e-10 (CNN ≈ 9e-5, Transformer ≈ 1.2). Buna rağmen LSTM'in test hatası CNN'den kötü değildir; sönümlenme, bu görevde uzun bağlamın zaten işe yaramadığı bulgusuyla birlikte okunmalıdır.
6. **Çıkarım gecikmesi (CPU, batch=1):** LSTM en hızlı (0.58 ms), CNN 1.10 ms, Transformer 2.04 ms. Üçü de gerçek zamanlı bir kovan izleme uygulaması için fazlasıyla hızlıdır.
7. **Few-shot transfer (Bad Schwartau):** Adaptasyonsuz modeller zaten ≈ 0.156–0.163 MAE vermektedir ve tüm yöntemler (sıfırdan, linear probe, fine-tuning) 0.156–0.20 aralığında kalmaktadır. Bu değerler trivial referanslarla (sıfır tahmin 0.1605, n-gün ortalaması 0.155–0.168) aynı düzeydedir; yani **hiçbir transfer yönteminin bu veriyle trivial referansı anlamlı biçimde geçtiği gösterilememiştir.** LSTM'de 30 günlük sıfırdan eğitim belirgin kötüleşmiştir (0.198), LSTM ve Transformer'da linear probe en kararlı seçenek olmuştur; CNN'de ise sıfırdan eğitim 7 günde en iyi sonucu vermiştir. Hedef kovan test kümesi yalnızca 248 örnek içerdiğinden bu sonuçlar kesin değil, yönlendiricidir.

**"Hangi koşulda hangi mimari?" sorusunun cevabı**

> Bu veri ölçeğinde (tek kaynak kovan, 48–50 bin parametre) **Causal CNN ve LSTM eşdeğer doğruluk verir; seçim doğruluğa değil dağıtım koşuluna göre yapılmalıdır.** LSTM en düşük gecikmeye ve en küçük bellek ayak izine sahiptir; CNN ise paralel eğitilebilir ve gradyanı geçmişe LSTM'den çok daha iyi taşır, uzun bağlamın gerçekten yararlı olduğu bir veriyle çalışılırsa avantajı belirginleşebilir. **Causal Transformer, bu küçük veri rejiminde ve uzun pencerede önerilmez**; zayıf endüktif yanlılığı nedeniyle varyansı yüksektir ve test dönemine genelleşmemiştir. Daha fazla kovan ve daha uzun kayıt olduğunda bu sonucun değişip değişmeyeceği bu çalışmada test edilmemiştir.

**Sınırlılıklar:** tek kaynak kovan ve tek test dönemi (2019 Şubat–Mayıs, bir mevsim); transfer kovanı yalnızca 12 saatlik çözünürlükte; tuning bütçesi küçük (6 deneme); modeller trivial referanslara göre sınırlı kazanç sağlıyor.
