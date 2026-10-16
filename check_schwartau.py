import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import data as D

df, nev = D.build_12h("schwartau")
print("rows", len(df), "range", df.index.min(), df.index.max())
raw = df.copy()
for c in ["weight", "t_in", "h_in"] + D.WX_COLS:
    print(f"  non-null {c:7s}: {int(df[c].notna().sum())}")
seg = df["seg"].values
for s in range(seg.max() + 1):
    pos = np.flatnonzero(seg == s)
    print(f"  seg {s}: {df.index[pos[0]]} -> {df.index[pos[-1]]}  steps={len(pos)}")
# where is weight missing?
m = df["weight"].isna()
print("weight NaN runs (start, end, n):")
start = None
for i, v in enumerate(m.values):
    if v and start is None: start = i
    if (not v) and start is not None:
        if i - start >= 6: print("  ", df.index[start], df.index[i - 1], i - start)
        start = None
hm = df["h_in"].isna()
print("humidity NaN total:", int(hm.sum()), " temp NaN total:", int(df['t_in'].isna().sum()))
w = df["weight"].dropna()
d = w.diff().abs()
print("12h |dw| quantiles 50/90/99:", d.quantile([.5, .9, .99]).round(3).tolist(), "max", round(d.max(), 2))
print(df["weight_raw"].describe().round(2).to_dict())
