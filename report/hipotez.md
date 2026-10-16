# Kovan Ağırlık Değişimi Tahmini - Hipotez Raporu (Aşama 2)
**Tarih:** 16 Ekim 2026  
**Problem:** Kovan sensör verileri (ağırlık, iç sıcaklık, iç nem) ve dış hava durumu verileri kullanılarak sonraki 24 saatteki ağırlık değişiminin ($\Delta \text{weight}_{t+24}$ - nektar akışı) tahmini.  
**Karşılaştırılan Mimariler:** Causal Dilated Residual CNN (TCN), LSTM, Causal Masked Transformer.

---

## 1. Endüktif Yanlılık (Inductive Bias) Çerçevesinde Mimari Analizi

Model performanslarını belirleyen en temel teorik unsur, seçilen mimarinin probleme özgü veri yapısına dair taşıdığı öncül varsayımlardır (endüktif yanlılık):

### A. Causal Dilated Residual CNN (TCN)
* **Endüktif Yanlılık:** **Yerellik (Locality)** ve **Zamansal Öteleme Eşdeğerliği (Temporal Translation Equivariance)**.
* **Kovan Dinamiklerine Uyumu:** Kovan içi metabolik aktiviteler ve nektar akışı 24 saatlik güneş döngüsüne ve periyodik hava hareketlerine sıkı sıkıya bağlıdır. CNN, çekirdek ağırlıklarını tüm zaman boyunca paylaşarak yerel desenleri (örn. sabah çıkışındaki sıcaklık-ağırlık ilişkisi) konumdan bağımsız olarak tanır. Dilation mekanizması sayesinde uzun geçmişi (haftalık 168 saati) hiyerarşik olarak sıkıştırmadan kavrar.
* **Beklenti:** Veri boyutunun küçük/orta ölçekli olduğu bu senaryoda aşırı öğrenmeye (overfitting) karşı en dirençli ve en kararlı model olacaktır.

### B. LSTM (Long Short-Term Memory)
* **Endüktif Yanlılık:** **Ardışıklık (Recurrence)** ve **Markovian Durum Güncellemesi (State-Space Bottleneck)**.
* **Kovan Dinamiklerine Uyumu:** Kovanın bir anlık biyolojik kütlesi ve koloninin gücü, geçmiş zaman adımlarının kümülatif birikimidir. LSTM’in gizli durumu ($h_t$) ve hücre hafızası ($c_t$), koloninin iç durumunu taşımak için doğal bir yapı sunar.
* **Kısıt ve Beklenti:** 168 adımlık uzun bağlamlarda BPTT sırasında gradyan sönümlenmesi (vanishing gradient) ve sabit boyutlu durum vektörünün uzun geçmişi özetlemede yaşayacağı bilgi kaybı (compression bottleneck) nedeniyle CNN'in biraz gerisinde kalması beklenmektedir.

### C. Causal Masked Transformer
* **Endüktif Yanlılık:** **Global İlişki ve Düşük Yapısal Yanlılık (Weak Inductive Bias)**.
* **Kovan Dinamiklerine Uyumu:** Transformer, adımlar arasındaki bağımsızlığı varsaymaz; dinamik self-attention ile herhangi iki saat arasındaki (örn. 5 gün önceki yağış ile bugünkü ağırlık) doğrudan korelasyonu yakalayabilir.
* **Kısıt ve Beklenti:** Konumsal öteleme değişmezliği doğuştan gelmediği (yalnızca positional encoding ile öğrenildiği) ve self-attention mekanizması yüksek veri hacmine ihtiyaç duyduğu için, sınırlı sayıdaki kovan verisinde aşırı öğrenmeye en yatkın model olacaktır.

---

## 2. Önceden Belirlenen Hipotez (Eğitim Öncesi Öngörü)

1. **Az ve Orta Ölçekli Veride:**  
   $$\text{Causal CNN} \ge \text{LSTM} > \text{Causal Transformer}$$
   Causal CNN, güçlü yerel yanlılığı ve dilation sayesinde en düşük test hatasını (MAE/RMSE) ve en dar güven aralığını verecektir.
2. **Bağlam Uzunluğu Arttıkça (1 Gün $\to$ 3 Gün $\to$ 7 Gün):**  
   Transformer'ın bağlam genişledikçe performans artışı LSTM'e kıyasla daha belirgin olacak; LSTM ise 168 adımda gradyan sönümlemesi nedeniyle doyuma ulaşacaktır.
3. **Çıkarım Hızı (Latency):**  
   Causal CNN paralelleştirilebilir konvolüsyon yapısıyla CPU üzerinde en düşük p50/p95 gecikmesini sunarken, Transformer matris çarpımları nedeniyle en yüksek gecikmeye sahip olacaktır.
