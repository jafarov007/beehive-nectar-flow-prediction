import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import data as D

src = D.source12_data()
print("source12: train", len(src["train"]), "val", len(src["val"]), "| y std train(kg/24h):", round(src["sd_y"], 3))
t = D.target12_data(src)
df = t["df"]
print("target: train_seg", t["train_seg"], df.index[np.flatnonzero(df.seg.values == t["train_seg"])[[0, -1]]].tolist())
print("target: test_seg ", t["test_seg"], df.index[np.flatnonzero(df.seg.values == t["test_seg"])[[0, -1]]].tolist(), "n_test", len(t["test"]))
for n in (2, 7, 30):
    e = t["train_ends"](n)
    print(f"  n={n:2d}g -> {len(e)} egitim ornegi, hedef zamani {df.index[e[0]+2]} .. {df.index[e[-1]+2]}")
y = t["y"][t["test"]]
print("target test y kg/24h: mean", y.mean().round(3), "std", y.std().round(3), "min", y.min().round(2), "max", y.max().round(2))
print("source test-less; source y std", src["sd_y"])
print("persistence MAE on target test:", np.mean(np.abs(y - t["persist"][t["test"]])).round(4), "| zero-pred MAE:", np.mean(np.abs(y)).round(4))
# normalised target-hive features (is the shift large?)
X = t["X"][t["test"]]
print("target test feature means (src-normalised):", dict(zip(D.FEATURES, X.mean(0).round(2))))
