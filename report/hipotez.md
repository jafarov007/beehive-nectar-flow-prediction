# Kovan Ağırlık Değişimi Tahmini — Eğitim Öncesi Hipotez ve Endüktif Yanlılık Raporu

**Tarih:** 16 Ekim 2026  
**Problem:** Kovan telemetri sensörleri ($w_t, T_{\text{in}}, H_{\text{in}}$) ve saatlik dış meteoroloji verileri ($T_{\text{out}}, H_{\text{out}}, P_{\text{out}}, V_{\text{wind}}$) kullanılarak sonraki 24 saatteki kovan ağırlığı değişiminin ($\Delta \text{weight}_{t+24} \in \mathbb{R}$, nektar akışı) kestirimi.  
**Karşılaştırılan Mimariler:** Causal Dilated Residual CNN (TCN), Long Short-Term Memory (LSTM), Causal Masked Transformer.

---

## 1. Endüktif Yanlılık (Inductive Bias) Çerçevesinde Teorik Mimari Analizi

Makine öğreniminde bir modelin genelleştirme kabiliyeti, hipotez uzayının ($\mathcal{H}$) problemin fiziksel ve biyolojik veri üretim sürecine uyumlu öncül varsayımlarla (endüktif yanlılık) ne ölçüde kısıtlandığına bağlıdır:

$$\hat{f} = \arg\min_{f \in \mathcal{H}} \mathcal{L}_{\text{train}}(f) + \Omega(f)$$

### A. Causal Dilated Residual CNN (TCN)
* **Temel Endüktif Yanlılık:** **Yerellik (Locality)** ve **Zamansal Öteleme Eşdeğerliği (Temporal Translation Equivariance)**:
  $$f(\mathcal{T}_\tau X) = \mathcal{T}_\tau f(X)$$
* **Biyolojik Kovan Dinamiklerine Uyumu:**  
  Bal arılarının (*Apis mellifera*) nektar toplama uçuşları, kovan içi havalandırma faaliyetleri ve metabolik tüketimi, doğrudan 24 saatlik güneş döngüsüne (sirkadiyen ritim) ve yerel sıcaklık değişimlerine bağlıdır. Konvolüsyonel çekirdekler zaman ekseni boyunca ağırlık paylaştığı için (weight sharing), günün belirli saatlerinde ortaya çıkan tipik mikro-örüntüleri (örn. sabah uçuş başlangıcındaki ani ağırlık kaybı veya akşam nektar depolaması) zamansal konumdan bağımsız olarak çok az serbest parametreyle öğrenebilir.
* **Genişletilmiş Geçmiş (Dilation):**  
  Üstel artan dilation faktörleri ($d \in \{1, 2, 4, 8, 16, 32\}$) sayesinde, 168 saatlik haftalık hava örüntüleri herhangi bir bilgi sıkıştırma darboğazına (compression bottleneck) uğramadan hiyerarşik olarak $RF=253$ adımlık bir kapsama alanında işlenir.
* **Ampirik Beklenti:**  
  Kısıtlı etiketli veri hacminde (~10.000 saatlik eğitim örneği) güçlü yapısal kısıtları sayesinde **aşırı öğrenmeye (overfitting) en dirençli** ve en düşük genelleştirme varyansına sahip mimari olması beklenmektedir.

### B. Long Short-Term Memory (LSTM)
* **Temel Endüktif Yanlılık:** **Ardışıklık (Strict Recurrence)** ve **Markovian Durum Güncellemesi**:
  $$h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})$$
* **Biyolojik Kovan Dinamiklerine Uyumu:**  
  Kovanın mevcut biyokütlesi, yavru alanı büyüklüğü ve bal stoku kümülatif birikimli bir durum değişkenidir (state variable). LSTM hücresinin gizli durumu ($h_t \in \mathbb{R}^H$) ve bellek hücresi ($c_t \in \mathbb{R}^H$), biyolojik koloninin içsel dinamik durumunu zamanda ileriye doğru taşımak için son derece doğal bir endüktif yanlılığa sahiptir.
* **Teorik Kısıt:**  
  Buna karşılık, 168 zaman adımı boyunca geriye yayılımda (BPTT), kapı gradyanlarının kümülatif çarpımı nedeniyle gradyan sönümlenmesi (vanishing gradient) kaçınılmazdır. Ayrıca sabit boyutlu ($H=60$) tek bir vektörün 7 günlük meteorolojik fırtına/kuraklık geçmişini sıkıştırma zorunluluğu bilgi kaybına yol açabilir.
* **Ampirik Beklenti:**  
  Kısa bağlamlarda çok güçlü bir performans sergilemesi, ancak bağlam 168 saate uzadığında doyuma ulaşarak CNN'in bir miktar gerisinde kalması beklenmektedir.

### C. Causal Masked Transformer
* **Temel Endüktif Yanlılık:** **Global İlişkisellik ve Zayıf Yapısal Yanlılık (Weak Inductive Bias)**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right) V, \quad M_{ij} = \begin{cases} 0, & i \ge j \\ -\infty, & i < j \end{cases}$$
* **Biyolojik Kovan Dinamiklerine Uyumu:**  
  Transformer mimarisi, adımlar arasında doğuştan gelen hiçbir yerellik veya ardışıklık varsaymaz. Dinamik dikkat matrisi sayesinde, örneğin 5 gün önceki şiddetli bir yağış ile bugünkü nektar açığı arasında doğrudan $\mathcal{O}(1)$ mesafeli bağ kurabilir.
* **Teorik Kısıt:**  
  Zaman serilerinde konumsal sıra bilgisi mimarinin içine gömülü değildir; yalnızca sinüzoidal konumsal kodlama ($PE$) ile modele sonradan eklenir. Zayıf endüktif yanlılığa sahip modellerin düzenlilikleri sıfırdan öğrenebilmesi için devasa veri hacimleri gereklidir.
* **Ampirik Beklenti:**  
  Tek bir kovanın kısıtlı verisinde, serbest parametrelerin aşırı esnekliği nedeniyle eğitim kümesine kolayca aşırı uyum sağlayacağı (overfitting) ve test kümesinde en yüksek genelleştirme hatasını ve en geniş güven aralığını üreteceği öngörülmektedir.

---

## 2. Eğitim Öncesi Hipotez Kararları (Nihai Sıralama Öngörüsü)

Eğitim boru hattı başlatılmadan önce aşağıdaki üç spesifik hipotez kayıt altına alınmıştır:

1. **Ana Performans Sıralaması (Kilitli Test Kümesi):**  
   $$\text{Causal Dilated CNN (TCN)} \ge \text{LSTM} > \text{Causal Transformer}$$
   *Gerekçe:* Yerellik ve öteleme eşdeğerliği, 24 saatlik sirkadiyen kovan döngüsünü modellemek için en güçlü koruyucu yanlılıktır.
2. **Bağlam Uzunluğuna Duyarlılık ($24\text{s} \to 72\text{s} \to 168\text{s}$):**  
   Transformer'ın bağlam genişledikçe göreceli hata düşüşü LSTM'e kıyasla daha belirgin olacak; LSTM ise 168 adımda gradyan sönümlenmesi ve durum sıkışması nedeniyle doyuma ulaşacaktır.
3. **Çıkarım Gecikmesi (CPU Latency, Batch=1):**  
   Paralel konvolüsyonel çekirdekler sayesinde Causal CNN en düşük çıkarım gecikmesini ($p_{50}, p_{95}$) sunacak; yoğun matris çarpımları nedeniyle Transformer en yüksek gecikmeye sahip olacaktır.
