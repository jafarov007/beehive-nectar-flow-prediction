"""
Veri katmani: GERCEK Kaggle "Beehive Metrics" arsivi + Open-Meteo dis hava verisi.

Kovanlar
  * wurzburg  : saatlik, ana (kaynak) kovan. Zamana gore train/val/test.
  * schwartau : agirlik/nem gunde 2 kez (12 saatte bir) -> YENI (transfer) kovan.
  * '2017' dosyalari wurzburg'un birebir alt kumesi oldugu icin KULLANILMAZ (sizinti olur).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather import fetch_weather  # noqa: E402

ARCHIVE = "C:/Users/serxan.m/Desktop/archive/"
FEATURES = ["dw", "t_in", "h_in", "t_out", "h_out", "precip", "wind", "hour_sin", "hour_cos"]
N_FEATURES = len(FEATURES)

EVENT_KG_1H = 2.0    # saatlik |dAgirlik| > 2 kg  -> mudahale/olay (hasat, kasa ekleme, sensor sifirlama)
EVENT_KG_12H = 3.0   # 12 saatlik adim icin
HORIZON_1H = 24      # 24 saat sonrasi
HORIZON_12H = 2      # 2 adim x 12 saat = 24 saat

TRAIN_END = pd.Timestamp("2018-05-31 23:00")
VAL_START = pd.Timestamp("2018-10-01")
VAL_END = pd.Timestamp("2019-02-23 23:00")
TEST_START = pd.Timestamp("2019-02-24")   # KILITLI son donem

_CACHE = {}


# ----------------------------------------------------------------- okuma / temizlik
def _series(name, col):
    df = pd.read_csv(f"{ARCHIVE}{col}_{name}.csv", parse_dates=["timestamp"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().set_index("timestamp")[col].sort_index()


def _clean(s, kind):
    if kind == "weight":
        if s.median() > 1000:      # gram -> kg
            s = s / 1000.0
        return s[s > 5]            # sensor hatasi (negatif / ~0) at
    if kind == "humidity":
        return s[(s > 0) & (s <= 100)]   # -100 sentinel degerlerini at
    return s[(s > -30) & (s < 60)]


def _segments(df, need):
    valid = df[need].notna().all(axis=1).values
    seg = np.full(len(df), -1, dtype=int)
    sid, prev = -1, False
    for i, v in enumerate(valid):
        if v:
            if not prev:
                sid += 1
            seg[i] = sid
        prev = v
    return seg


def _correct_events(df, thr):
    """Her kesintisiz segment icinde |dw|>thr olan sicramalari 0'lar ve agirligi yeniden insa eder
    (veriyi atmadan seviye kaymasini yok eder)."""
    w = df["weight"].values.copy()
    seg = df["seg"].values
    wc = np.full(len(df), np.nan)
    dw = np.zeros(len(df))
    nev = 0
    for sid in range(seg.max() + 1):
        pos = np.flatnonzero(seg == sid)
        d = np.diff(w[pos], prepend=w[pos][0])
        ev = np.abs(d) > thr
        nev += int(ev.sum())
        d[ev] = 0.0
        wc[pos] = w[pos][0] + np.cumsum(d)
        dw[pos] = d
    df["weight_raw"] = df["weight"]
    df["weight"] = wc
    df["dw"] = dw
    return nev


def _time_feats(df):
    hr = df.index.hour.values
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)


WX_RENAME = {"out_temp": "t_out", "out_hum": "h_out", "out_precip": "precip", "out_wind": "wind"}
WX_COLS = list(WX_RENAME.values())


# ----------------------------------------------------------------- saatlik (wurzburg)
def build_hourly(name="wurzburg"):
    key = ("h", name)
    if key in _CACHE:
        return _CACHE[key]
    w = _clean(_series(name, "weight"), "weight").resample("1h").last()
    t = _clean(_series(name, "temperature"), "temperature").resample("1h").mean()
    h = _clean(_series(name, "humidity"), "humidity").resample("1h").mean()
    df = pd.concat([w, t, h], axis=1, keys=["weight", "t_in", "h_in"], sort=True)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1h"))
    for c in ["weight", "t_in", "h_in"]:
        df[c] = df[c].interpolate(limit=3, limit_area="inside")
    wx = fetch_weather(name).rename(columns=WX_RENAME)
    df = df.join(wx, how="left")
    df["seg"] = _segments(df, ["weight", "t_in", "h_in"] + WX_COLS)
    nev = _correct_events(df, EVENT_KG_1H)
    _time_feats(df)
    _CACHE[key] = (df, nev)
    return df, nev


# ----------------------------------------------------------------- 12 saatlik
def _agg12(s, how="mean"):
    r = s.resample("12h", offset="1h", closed="right", label="right")
    return r.sum() if how == "sum" else r.mean()


def _wx12(name):
    wx = fetch_weather(name).rename(columns=WX_RENAME)
    out = pd.DataFrame({
        "t_out": _agg12(wx["t_out"]), "h_out": _agg12(wx["h_out"]),
        "precip": _agg12(wx["precip"], "sum"), "wind": _agg12(wx["wind"]),
    })
    return out


def build_12h(name):
    key = ("12", name)
    if key in _CACHE:
        return _CACHE[key]
    if name == "wurzburg":
        h, _ = build_hourly("wurzburg")
        hv = h[h["seg"] >= 0]
        cnt = hv["weight"].resample("12h", offset="1h", closed="right", label="right").count()
        df = pd.DataFrame({
            "weight": _agg12(hv["weight"]), "t_in": _agg12(hv["t_in"]), "h_in": _agg12(hv["h_in"]),
        })
        df.loc[cnt < 10, ["weight", "t_in", "h_in"]] = np.nan
    else:  # schwartau: agirlik & nem zaten 12 saatlik; sicaklik 5 dakikalik -> 12h ort.
        w = _clean(_series(name, "weight"), "weight")
        h = _clean(_series(name, "humidity"), "humidity")
        t = _agg12(_clean(_series(name, "temperature"), "temperature"))
        df = pd.concat([w, t, h], axis=1, keys=["weight", "t_in", "h_in"], sort=True)
        df = df[~df["weight"].isna() | ~df["h_in"].isna() | ~df["t_in"].isna()]
    wx = _wx12(name)
    start, end = df.index.min(), df.index.max()
    grid = pd.date_range(start, end, freq="12h")
    df = df.reindex(grid).join(wx, how="left")
    df["seg"] = _segments(df, ["weight", "t_in", "h_in"] + WX_COLS)
    nev = _correct_events(df, EVENT_KG_12H)
    _time_feats(df)
    _CACHE[key] = (df, nev)
    return df, nev


# ----------------------------------------------------------------- orneklem uretimi
def make_idx(seg, W, h):
    """Penceresi ve hedefi AYNI kesintisiz segment icinde kalan bitis konumlari."""
    out = []
    for sid in np.unique(seg[seg >= 0]):
        pos = np.flatnonzero(seg == sid)
        a, b = pos[0] + W - 1, pos[-1] - h
        if b >= a:
            out.append(np.arange(a, b + 1))
    return np.concatenate(out) if out else np.array([], dtype=int)


def feature_stats(df, mask):
    rows = df.loc[mask & (df["seg"] >= 0), FEATURES]
    mu = rows.mean().values.astype(np.float32)
    sd = rows.std().replace(0, 1.0).values.astype(np.float32)
    return mu, sd


def normalise(df, mu, sd):
    X = ((df[FEATURES].values - mu) / sd).astype(np.float32)
    return np.nan_to_num(X, nan=0.0)


def _targets(df, h):
    w = df["weight"].values
    y = np.full(len(df), np.nan)
    y[:-h] = w[h:] - w[:-h]
    return y


def hourly_data(W, train_stride=2):
    """wurzburg: zamana gore bolunmus penceli veri. Normalizasyon SADECE train'den."""
    df, _ = build_hourly("wurzburg")
    idx_all = make_idx(df["seg"].values, W, HORIZON_1H)
    times = df.index.values
    t_idx = pd.DatetimeIndex(times[idx_all])
    tr = idx_all[t_idx <= TRAIN_END]
    va = idx_all[(t_idx >= VAL_START) & (t_idx <= VAL_END)]
    te = idx_all[t_idx >= TEST_START]
    mu, sd = feature_stats(df, df.index <= TRAIN_END)
    X = normalise(df, mu, sd)
    y = _targets(df, HORIZON_1H)
    mu_y, sd_y = float(y[tr].mean()), float(y[tr].std())
    wc = df["weight"].values
    persist = np.full(len(df), np.nan)
    if W >= 25:
        persist[24:] = wc[24:] - wc[:-24]
    return {
        "X": X, "y": y, "W": W, "mu_y": mu_y, "sd_y": sd_y,
        "train": tr[::train_stride], "train_full": tr, "val": va, "test": te,
        "persist": persist, "df": df,
    }


# ----------------------------------------------------------------- transfer: 12 saatlik cozunurluk
def source12_data(W=14, train_stride=1):
    """Kaynak: wurzburg 12h (train<=TRAIN_END, val). Transfer kovani ASLA burada kullanilmaz."""
    df, _ = build_12h("wurzburg")
    idx_all = make_idx(df["seg"].values, W, HORIZON_12H)
    t_idx = pd.DatetimeIndex(df.index.values[idx_all])
    tr = idx_all[t_idx <= TRAIN_END]
    va = idx_all[(t_idx >= VAL_START) & (t_idx <= VAL_END)]
    mu, sd = feature_stats(df, df.index <= TRAIN_END)
    y = _targets(df, HORIZON_12H)
    return {
        "X": normalise(df, mu, sd), "y": y, "W": W, "mu_y": float(y[tr].mean()), "sd_y": float(y[tr].std()),
        "train": tr[::train_stride], "train_full": tr, "val": va, "test": np.array([], dtype=int),
        "mu": mu, "sd": sd, "df": df,
    }


def target12_data(src, W=14, n_max=30):
    """Yeni kovan (schwartau). Normalizasyon KAYNAK istatistikleriyle.
    Egitim: kronolojik olarak ilk uzun segmentin basindan n gun (once W adim girdi-gecmisi/isinma).
    Test  : sonraki (daha ileri tarihli) segmentin tamami -> tum n degerleri icin AYNI test dilimi."""
    df, _ = build_12h("schwartau")
    seg = df["seg"].values
    lens = {s: int((seg == s).sum()) for s in range(seg.max() + 1)}
    need_tr = W + 2 * n_max + HORIZON_12H
    tr_seg = next(s for s in sorted(lens) if lens[s] >= need_tr)
    te_seg = max(s for s in lens if s > tr_seg and lens[s] >= W + HORIZON_12H + 50)
    pos = np.flatnonzero(seg == tr_seg)
    t0 = pos[0] + W - 1
    y = _targets(df, HORIZON_12H)
    wc = df["weight"].values
    te_mask = np.where(seg == te_seg, seg, -1)
    test = make_idx(te_mask, W, HORIZON_12H)
    persist = np.full(len(df), np.nan)
    persist[HORIZON_12H:] = wc[HORIZON_12H:] - wc[:-HORIZON_12H]
    return {
        "X": normalise(df, src["mu"], src["sd"]), "y": y, "W": W, "mu_y": src["mu_y"], "sd_y": src["sd_y"],
        "test": test, "persist": persist, "df": df, "train_seg": tr_seg, "test_seg": te_seg,
        # n gun etiketli veri: bitis t0..t0+2n-2  (hedef t+2 <= t0+2n)
        "train_ends": lambda n: np.arange(t0, t0 + 2 * n - 1),
    }



# ----------------------------------------------------------------- batch iterator
def iterate(data, split_idx, bs, shuffle, rng=None, W=None):
    import torch
    W = W or data["W"]
    X, y = data["X"], data["y"]
    ar = np.arange(W)
    order = rng.permutation(len(split_idx)) if shuffle else np.arange(len(split_idx))
    for i in range(0, len(order), bs):
        b = split_idx[order[i:i + bs]]
        win = X[b[:, None] - (W - 1) + ar]
        yn = ((y[b] - data["mu_y"]) / data["sd_y"]).astype(np.float32)
        yield torch.from_numpy(win), torch.from_numpy(yn)


def set_seed(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df, nev = build_hourly("wurzburg")
    print(f"[wurzburg 1h] satir={len(df)}  gecerli={(df.seg >= 0).sum()}  segment={df.seg.max() + 1}  olay(|dw|>{EVENT_KG_1H}kg)={nev}")
    for W in (24, 72, 168):
        d = hourly_data(W)
        print(f"  W={W:3d}: train={len(d['train_full'])} (stride2: {len(d['train'])})  val={len(d['val'])}  test={len(d['test'])}")
    d = hourly_data(168)
    yt = d["y"][d["test"]]
    print(f"  test hedef (kg/24h): ort={yt.mean():.3f} std={yt.std():.3f} min={yt.min():.2f} max={yt.max():.2f}")
    ytr = d["y"][d["train_full"]]
    print(f"  train hedef (kg/24h): ort={ytr.mean():.3f} std={ytr.std():.3f}")
    for nm in ("wurzburg", "schwartau"):
        d12, nev12 = build_12h(nm)
        seg = d12["seg"].values
        lens = sorted([int((seg == s).sum()) for s in range(seg.max() + 1)], reverse=True)[:5]
        print(f"[{nm} 12h] satir={len(d12)} gecerli={(seg >= 0).sum()} segment={seg.max() + 1} en uzun={lens} olay(|dw|>{EVENT_KG_12H}kg)={nev12}")
