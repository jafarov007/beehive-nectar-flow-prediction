import sys, os, time
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, torch, torch.nn as nn
import data as D
from models import make_model
import train as T

print("cpu_count:", os.cpu_count())
torch.set_num_threads(2)
data = D.hourly_data(168)
hp = {"lr": 2e-3, "dropout": 0.1, "wd": 1e-4}
rng = np.random.default_rng(0)

for arch in ["cnn", "lstm", "transformer"]:
    D.set_seed(0)
    m = make_model(arch, 168, hp["dropout"])
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"])
    crit = nn.HuberLoss()
    t0 = time.time(); n = 0
    m.train()
    for X, y in D.iterate(data, data["train"], 64, True, rng):
        opt.zero_grad(); crit(m(X), y).backward(); opt.step(); n += 1
    dt = time.time() - t0
    print(f"{arch:12s} 1 epoch (train stride2, {n} batch): {dt:.1f}s  -> val eval:", end=" ")
    t0 = time.time(); T.evaluate(m, data, data["val"]); print(f"{time.time()-t0:.1f}s")

# overfit testi: 64 ornegi ezberleyebiliyor mu? (test setine dokunmaz)
print("\nOverfit testi (64 train ornegi, 150 adim):")
sub = data["train"][:64]
for arch in ["cnn", "lstm", "transformer"]:
    D.set_seed(0)
    m = make_model(arch, 168, 0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0)
    X, y = next(D.iterate(data, sub, 64, False))
    for step in range(150):
        m.train(); opt.zero_grad(); l = nn.functional.mse_loss(m(X), y); l.backward(); opt.step()
    print(f"  {arch:12s} final MSE (normalised) = {l.item():.5f}   [baslangic varyansi ~ {float(y.var()):.3f}]")
