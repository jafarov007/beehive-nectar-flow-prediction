# Kovan Ağırlık Değişimi Tahmini (Proje Dokümantasyonu)

Bu proje, **Ahmet Hoca'nın Ödevi** ve **Adım Adım Çalışma Planı** doğrultusunda hazırlanmış eksiksiz bir Zaman Serisi Modellemesi ve Derin Öğrenme projesidir.

---

## 📁 Proje Dosya ve Klasör Yapısı

```
C:\Users\serxan.m\Desktop\kovan_project\
│
├── data/
│   └── beehive_metrics.csv       # 15 dakikalık kovan içi/dışı sensör ve ağırlık verisi (3 kovan)
│
├── src/
│   ├── data.py                   # Saatlik resample, anomali temizliği, özellik mühendisliği, zamansal bölme
│   ├── models.py                 # Causal CNN (TCN), LSTM, Causal Transformer (~50k iso-parametre)
│   ├── train.py                  # 5-seed döngüsü, AdamW + Huber Loss, erken durdurma, %95 güven aralığı
│   └── analysis.py               # Pencere boyutu (1/3/7 gün), LSTM gradyan sönümleme, Latency, Transfer öğrenme
│
├── report/
│   ├── hipotez.md                # Aşama 2: Eğitimden önce yazılan Endüktif Yanlılık (Inductive Bias) metni
│   ├── tasarim_ve_hesaplamalar.md# Aşama 3: Receptive Field (RF=253 >= 168) ve elle parametre/MAC formülleri
│   └── sonuc_ve_rapor.md         # Aşama 8: Teslime hazır final rapor ve sonuç sentez paragrafı
│
├── results/                      # Üretilen CSV tabloları ve grafikler (.png)
│
├── run_all.py                    # Bütün aşamaları tek tıkla baştan sona çalıştıran ana betik
└── README.md                     # Proje kılavuzu
```

---

## 🚀 Nasıl Çalıştırılır?

1. **Terminali Açın:**  
   Proje dizinine geçin:
   ```powershell
   cd "C:\Users\serxan.m\Desktop\kovan_project"
   ```

2. **Tüm Pipeline'ı Çalıştırmak İçin:**
   ```powershell
   python run_all.py
   ```

3. **Tekil Modülleri Çalıştırmak İçin:**
   * **Modellerin parametre sayılarını görmek için:**
     ```powershell
     python src/models.py
     ```
   * **5-Seed Karşılaştırmalı Eğitimi başlatmak için:**
     ```powershell
     python src/train.py
     ```
   * **Analiz grafikleri ve transfer öğrenme testleri için:**
     ```powershell
     python src/analysis.py
     ```

---

## 📊 Ödev Maddelerinin Karşılanma Durumu

| Ödev Maddesi | İlgili Dosya / Çıktı | Açıklama |
| :--- | :--- | :--- |
| **1. Hipotez** | `report/hipotez.md` | Eğitimden önce endüktif yanlılık terimleriyle 3 mimari karşılaştırıldı. |
| **2. Tasarım & Receptive Field** | `report/tasarim_ve_hesaplamalar.md` | $RF = 253 \ge 168$ türetildi, parametre ve MAC formülleri elle çıkarıldı. |
| **3. İzo-Parametre** | `src/models.py` | Modeller ~48k–50k parametreye kalibre edildi (CNN: 48.7k, LSTM: 48.3k, Trans: 49.8k). |
| **4. 5 Seed & Güven Aralığı** | `results/main_*.json` & `results/fig_main.png` | 5 farklı seed (0-4) ile eğitildi, $t$-dağılımı ile %95 güven aralığı hesaplandı. |
| **5. Bağlam Penceresi (1, 3, 7 Gün)** | `results/windows_*.json` & `results/fig_window.png` | 24, 72 ve 168 saatlik girdi pencerelerinin test hatası grafiğe döküldü. |
| **6. LSTM Gradyan Normu** | `results/grad_norm.json` & `results/fig_grad.png` | BPTT sırasında geriye doğru zaman adımlarında gradyan sönümlenmesi çizildi. |
| **7. Çıkarım Gecikmesi (Latency)** | `results/latency.json` | CPU üzerinde tekil çıkarımda Median ve p95 milisaniye gecikmeleri ölçüldü. |
| **8. Transfer Öğrenme** | `results/transfer_*.json` & `results/fig_transfer.png` | $n \in \{2, 7, 30\}$ gün için Sıfırdan, Linear Probe ve Fine-Tuning kıyaslandı. |
| **9. Sonuç Bölümü & Rapor** | `report/sonuc_ve_rapor.md` & `report/Kovan_Rapor.docx` | Tüm tablolar, grafikler ve "Hangi koşulda hangi mimari, neden?" sentezi. |
