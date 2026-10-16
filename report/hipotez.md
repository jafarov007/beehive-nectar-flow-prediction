# Kovan Ağırlık Değişimi Tahmini — Eğitim Öncesi Hipotez ve Endüktif Yanlılık Raporu

**Öğrenci:** Sarkan MAMMADOV  
**Ders:** BM 618 - Derin Öğrenme ve Büyük Dil Modelleri  
**Öğretim Üyesi:** Dr. Öğr. Üyesi Ahmet ALBAYRAK  
**Tarih:** 16 Ekim 2026  
**Problem:** Kovan içi telemetri sensörleri ($w_t, T_{\text{in}}, H_{\text{in}}$) ve saatlik dış meteoroloji verileri ($T_{\text{out}}, H_{\text{out}}, P_{\text{out}}, V_{\text{wind}}$) kullanılarak sonraki 24 saatteki net kovan ağırlığı değişiminin ($\Delta \text{weight}_{t+24} \in \mathbb{R}$, nektar akışı) kestirimi.  
**Karşılaştırılan Mimariler:** Causal Dilated Residual CNN (TCN), Long Short-Term Memory (LSTM), Causal Masked Transformer.

---

## 1. Endüktif Yanlılık (Inductive Bias) Çerçevesinde Teorik Mimari Analizi

Makine öğreniminde bir modelin sınırlı veri altında sergileyeceği genelleştirme kabiliyeti, hipotez uzayının ($\mathcal{H}$) problemin fiziksel ve biyolojik veri üretim sürecine uyumlu yapısal öncül varsayımlarla ne ölçüde sınırlandırıldığına bağlıdır:

$$\hat{f} = \arg\min_{f \in \mathcal{H}} \mathcal{L}_{\text{train}}(f) + \Omega(f)$$

Burada düzenlileştirici terim $\Omega(f)$, doğrudan mimarinin endüktif yanlılığı (inductive bias) tarafından belirlenir. Kovan sensör telemetrisi ve meteorolojik zaman serileri, bal arılarının sirkadiyen biyolojisi ve yerel mikro-iklim koşullarıyla şekillenen periyodik dinamikler içerdiğinden, karşılaştırılan üç temel mimarinin kuramsal uygunluğu eğitim öncesinde şu şekilde modellenmiştir:

### 1.1 Causal Dilated Residual CNN (TCN)

Causal Dilated CNN mimarisinin temel yapısal yanlılığı, zaman ekseni üzerindeki katı **yerellik (locality)** ve **zamansal öteleme eşdeğerliği (temporal translation equivariance)** varsayımlarına dayanır:

$$f(\mathcal{T}_\tau X) = \mathcal{T}_\tau f(X)$$

Bal arılarının (*Apis mellifera*) nektar ve polen toplama uçuşları, kovan içi havalandırma faaliyetleri ve metabolik tüketimi doğrudan 24 saatlik sirkadiyen güneş döngüsüne ve yerel sıcaklık eğrilerine bağlıdır. Konvolüsyonel filtreler zaman ekseni boyunca ağırlık paylaştığından (weight sharing), günün belirli saatlerinde tekrarlayan mikro-örüntüleri (örneğin sabah uçuş başlangıcındaki ani kütle kaybı veya akşam saatlerindeki nektar depolaması) zamansal konumdan bağımsız olarak az sayıda serbest parametreyle öğrenebilir. 

Ayrıca üstel olarak genişleyen dilation hiyerarşisi ($d_i \in \{1, 2, 4, 8, 16, 32\}$), modelin 168 saatlik haftalık hava dalgalarını herhangi bir bilgi sıkıştırma darboğazına (compression bottleneck) uğramadan doğrudan $RF=253$ adımlık bir kapsama alanında işlemesini sağlar. Kısıtlı etiketli veri hacminde (~10.000 saatlik eğitim örneği) bu güçlü yapısal kısıtlar, Causal CNN'in aşırı öğrenmeye (overfitting) en dirençli ve en düşük genelleştirme varyansına sahip mimari olacağını düşündürmektedir.

### 1.2 Long Short-Term Memory (LSTM)

LSTM mimarisi, katı bir **ardışıklık (recurrence)** ve **Markovian gizli durum sıkıştırması** endüktif yanlılığı üzerine kuruludur:

$$h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})$$

Biyolojik açıdan bir arı kovanının anlık kütlesi, yavru alanı büyüklüğü ve bal stoğu kümülatif birikimli bir durum değişkenidir (state variable). LSTM hücresinin gizli durumu ($h_t \in \mathbb{R}^H$) ve bellek hücresi ($c_t \in \mathbb{R}^H$), biyolojik koloninin içsel durumunu zamanda ileriye doğru taşımak için probleme son derece uyumlu bir fiziksel temsil sunar.

Buna karşın, 168 zaman adımı boyunca geriye yayılımda (BPTT), kapı gradyanlarının kümülatif çarpımı nedeniyle gradyan sönümlenmesi (vanishing gradient) kaçınılmazdır. Ayrıca sabit boyutlu ($H=60$) tek bir gizli vektörün 7 günlük çok değişkenli meteorolojik geçmişi ve sirkadiyen döngüleri sıkıştırma zorunluluğu bilgi kaybına yol açabilir. Bu kuramsal sınırlar nedeniyle, LSTM'in kısa bağlamlarda güçlü bir kestirim performansı sergileyeceği, ancak bağlam 168 saate uzadığında doyuma ulaşarak CNN'in bir miktar gerisinde kalacağı öngörülmektedir.

### 1.3 Causal Masked Transformer

Transformer mimarisi, adımlar arasında doğuştan gelen hiçbir yerellik veya ardışıklık varsayımı içermez; **global dinamik ikili dikkat (pairwise self-attention)** ve **zayıf yapısal yanlılık (weak inductive bias)** taşır:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right) V, \quad M_{ij} = \begin{cases} 0, & i \ge j \\ -\infty, & i < j \end{cases}$$

Kuramsal açıdan dinamik dikkat mekanizması, adımlar arasındaki zamansal mesafeden bağımsız olarak $\mathcal{O}(1)$ yol uzunluğuyla doğrudan ilişkisellik kurabilir. Örneğin 5 gün önceki şiddetli bir fırtına ile bugünkü nektar açığı arasında doğrudan bağlantı kurulabilmektedir.

Ancak zaman serilerinde konumsal sıra bilgisi mimarinin içine gömülü olmayıp yalnızca sinüzoidal konumsal kodlama ($PE$) ile modele sonradan eklenmektedir. Zayıf endüktif yanlılığa sahip modellerin düzenlilikleri ve periyodik dinamikleri sıfırdan öğrenebilmesi için çok yüksek veri hacmi gerekmektedir. Tek bir kovanın sınırlı saatlik verisinde serbest parametrelerin aşırı esnekliği, modelin eğitim kümesine kolayca aşırı uyum sağlayacağını (overfitting) ve test kümesinde en yüksek genelleştirme hatasını ve en geniş güven aralığını üreteceğini düşündürmektedir.

---

## 2. Eğitim Öncesi Hipotez Kararları (Nihai Sıralama Öngörüsü)

Eğitim boru hattı başlatılmadan önce aşağıdaki üç spesifik hipotez kayıt altına alınmıştır:

1. **Ana Performans Sıralaması (Kilitli Test Kümesi):**  
   $$\text{Causal Dilated CNN (TCN)} \ge \text{LSTM} > \text{Causal Transformer}$$  
   *Gerekçe:* Yerellik ve zamansal öteleme eşdeğerliği, 24 saatlik sirkadiyen kovan döngüsünü modellemek için en güçlü koruyucu yanlılıktır.
2. **Bağlam Uzunluğuna Duyarlılık ($24\text{s} \to 72\text{s} \to 168\text{s}$):**  
   Transformer'ın bağlam genişledikçe göreceli hata düşüşü LSTM'e kıyasla daha belirgin olacak; LSTM ise 168 adımda gradyan sönümlenmesi ve durum sıkışması nedeniyle doyuma ulaşacaktır.
3. **Çıkarım Gecikmesi (CPU Latency, Batch=1):**  
   Paralel konvolüsyonel çekirdekler sayesinde Causal CNN en düşük çıkarım gecikmesini ($p_{50}, p_{95}$) sunacak; yoğun matris çarpımları nedeniyle Transformer en yüksek gecikmeye sahip olacaktır.
