# Aşama 3 — Tasarım: Receptive Field, Parametre ve MAC (elle hesap)

> Aşağıdaki **tüm sayılar önce elle türetilmiş, sonra `src/models.py` içindeki kodla doğrulanmıştır**
> (`python src/models.py`). Kod çıktısı elle hesapla birebir aynıdır.

Girdi: $T$ = dizi uzunluğu (7 gün, saatlik veri → $T=168$), $F=9$ özellik
(`dw`, `t_in`, `h_in`, `t_out`, `h_out`, `precip`, `wind`, `hour_sin`, `hour_cos`).

---

## 1. Receptive field türetimi

Saatlik veride: günlük döngü = 24 adım, haftalık hava örüntüsü = $7\times24 = 168$ adım.
Hedef: $RF \ge 168$ (günlük döngü için en az 24).

Bir residual blokta iki dilated conv (çekirdek $k$, dilation $d_i$) vardır. Her conv, RF'ye
$(k-1)d_i$ ekler. Bu yüzden $L$ blok için:

$$RF = 1 + 2\,(k-1)\sum_{i=1}^{L} d_i$$

Seçim: $k=3$, dilation $=\{1,2,4,8,16,32\}$ ($L=6$), $\sum d_i = 63$:

$$RF = 1 + 2\cdot2\cdot63 = 253 \ge 168 \;\checkmark$$

Bir eksik blokla ($\{1,\dots,16\}$, $\sum d=31$) $RF = 125 < 168$ olurdu; yani 6 blok gereklidir.

**Doğrulama:** son çıktı adımının girdiye gradyanı alınıp sıfırdan farklı olan en eski adım bulundu:
ölçülen RF $=253$ (kuramsal ile aynı).

Bağlam penceresi analizi için CNN, pencereye göre ayarlandı (RF $\ge W$ olacak şekilde):

| Pencere $W$ | Dilation | $\sum d$ | $RF=1+4\sum d$ | Kanal $C$ |
|---|---|---|---|---|
| 24 (1 gün) | 1,2,4 | 7 | 29 | 51 |
| 72 (3 gün) | 1,2,4,8,16 | 31 | 125 | 40 |
| 168 (7 gün) | 1,2,4,8,16,32 | 63 | 253 | 36 |

Kanal sayısı, her pencerede parametre sayısı ≈ 49–50 k kalsın diye değiştirildi.

---

## 2. Causal + dilated + residual CNN

* Her convdan önce sola $(k-1)d$ sıfır padding, sağdan kırpma yok ⇒ $y_t$ yalnızca $x_{\le t}$'ye bağlıdır.
* Blok: `Conv(dilated) → ReLU → Dropout → Conv(dilated) → ReLU → Dropout`, çıktı girdiyle toplanır
  (kanal sayısı aynı olduğu için 1×1 conv gerekmez).
* Giriş: $1\times1$ conv ($F\to C$). Çıkış: son zaman adımı → `Linear(C,32) → ReLU → Linear(32,1)`.

---

## 3. Elle parametre sayısı

Formüller (bias dahil): Conv1d $C_{in}C_{out}k + C_{out}$ · Linear $in\cdot out + out$ ·
LSTM (1 katman) $4\,[H(I+H) + 2H]$ (PyTorch'ta $b_{ih}$ ve $b_{hh}$ ayrı) ·
Transformer katmanı $= \underbrace{3d^2+3d}_{QKV} + \underbrace{d^2+d}_{W_O} + \underbrace{d\,d_{ff}+d_{ff}}_{FFN_1} + \underbrace{d_{ff}d+d}_{FFN_2} + \underbrace{4d}_{2\times LN}$.

Ortak baş: `Linear(h,32)` ($32h+32$) + `Linear(32,1)` ($33$).

### Causal CNN ($C=36$, 6 blok)
| Katman | Hesap | Parametre |
|---|---|---|
| giriş 1×1 conv | $9\cdot36+36$ | 360 |
| 1 conv ($k=3$) | $36\cdot36\cdot3+36$ | 3 924 |
| 1 blok (2 conv) | $2\cdot3924$ | 7 848 |
| 6 blok | $6\cdot7848$ | 47 088 |
| baş | $(36\cdot32+32)+33$ | 1 217 |
| **Toplam** | $360+47088+1217$ | **48 665** |

### LSTM ($H=60$, 2 katman)
| Katman | Hesap | Parametre |
|---|---|---|
| LSTM-1 ($I=9$) | $4\,[60\cdot69+120]$ | 17 040 |
| LSTM-2 ($I=60$) | $4\,[60\cdot120+120]$ | 29 280 |
| baş | $(60\cdot32+32)+33$ | 1 985 |
| **Toplam** | | **48 305** |

### Causal Transformer ($d=44$, $d_{ff}=88$, 3 katman, 4 kafa)
| Katman | Hesap | Parametre |
|---|---|---|
| giriş projeksiyonu | $9\cdot44+44$ | 440 |
| QKV | $3\cdot44^2+3\cdot44$ | 5 940 |
| $W_O$ | $44^2+44$ | 1 980 |
| FFN$_1$ | $44\cdot88+88$ | 3 960 |
| FFN$_2$ | $88\cdot44+44$ | 3 916 |
| 2×LayerNorm | $4\cdot44$ | 176 |
| 1 katman | | 15 972 |
| 3 katman | | 47 916 |
| baş | $(44\cdot32+32)+33$ | 1 473 |
| **Toplam** | $440+47916+1473$ | **49 829** |

Konum kodlaması sabit sinüzoidaldir (öğrenilen parametresi yok).

**Üç model arasındaki fark ≤ %3.1** (48 305 – 49 829): iso-parametre koşulu sağlandı.

---

## 4. Elle MAC sayısı ($T=168$, bir dizi için; bias, aktivasyon, LayerNorm sayılmadı)

Formüller: Conv1d $C_{in}C_{out}k\,T$ · Linear (tüm adımlara uygulanırsa) $in\cdot out\cdot T$ ·
LSTM $4H(I+H)\,T$ · Self-attention $4Td^2 + 2T^2 d$ (QKV + $W_O$ = $4Td^2$; $QK^\top$ ve $AV$ = $2T^2d$).

### Causal CNN
| Katman | Hesap | MAC |
|---|---|---|
| giriş 1×1 | $9\cdot36\cdot168$ | 54 432 |
| 1 conv | $36\cdot36\cdot3\cdot168$ | 653 184 |
| 12 conv (6 blok) | $12\cdot653184$ | 7 838 208 |
| baş (yalnız son adım) | $36\cdot32+32\cdot1$ | 1 184 |
| **Toplam** | | **7 893 824** |

### LSTM
| Katman | Hesap | MAC |
|---|---|---|
| LSTM-1 | $4\cdot60\cdot69\cdot168$ | 2 782 080 |
| LSTM-2 | $4\cdot60\cdot120\cdot168$ | 4 838 400 |
| baş | $60\cdot32+32$ | 1 952 |
| **Toplam** | | **7 622 432** |

### Causal Transformer
| Katman | Hesap | MAC |
|---|---|---|
| giriş projeksiyonu | $9\cdot44\cdot168$ | 66 528 |
| QKV + $W_O$ | $4\cdot168\cdot44^2$ | 1 300 992 |
| $QK^\top$ + $AV$ | $2\cdot168^2\cdot44$ | 2 483 712 |
| FFN | $2\cdot168\cdot44\cdot88$ | 1 300 992 |
| 1 katman | | 5 085 696 |
| 3 katman | | 15 257 088 |
| baş | $44\cdot32+32$ | 1 440 |
| **Toplam** | | **15 325 056** |

| Model | Parametre | MAC ($T=168$) | MAC / CNN |
|---|---|---|---|
| Causal CNN | 48 665 | 7.89 M | 1.00 |
| LSTM | 48 305 | 7.62 M | 0.97 |
| Causal Transformer | 49 829 | 15.33 M | 1.94 |

> [!NOTE]
> Causal mask nedeniyle dikkat matrisinin yarısı kullanılmaz; yukarıdaki sayı **yoğun (dense)
> hesaplama** içindir, çünkü PyTorch tam $T\times T$ matrisini hesaplayıp maskeler.
> MAC yarışında LSTM ile CNN eşit, Transformer yaklaşık 2× pahalıdır; ancak LSTM'in MAC'leri
> zamanda **sıralıdır** (paralelleştirilemez), bu yüzden MAC gerçek gecikmeyi tek başına belirlemez.

**Doğrulama kodu:** `count_macs()` (Conv1d/Linear/LSTM/MultiheadAttention modüllerinin gerçek
tensör şekillerinden, forward hook ile) ve `sum(p.numel())` yukarıdaki tüm toplamlarla birebir eşleşti.
