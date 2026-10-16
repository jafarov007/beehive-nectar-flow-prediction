"""
Analiz asamalari:
  windows  : girdi penceresi 1/3/7 gun (24/72/168 saat) -> test MAE (5 seed)
  grad     : girdi gradyan normunun zamana gore degisimi (LSTM; CNN & Transformer ile kiyas)
  latency  : batch=1, CPU, median & p95
  transfer : yeni kovan (schwartau, 12 saatlik cozunurluk) n=2/7/30 gun; scratch / linear probe / fine-tune
Kullanim:  python src/analysis.py windows --arch cnn
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D                                              # noqa: E402
from models import make_model, ARCHS, count_params           # noqa: E402
import train as T                                             # noqa: E402

RESULTS = T.RESULTS


def best_hp(arch):
    return json.load(open(os.path.join(RESULTS, f"main_{arch}.json")))["best_hp"]


# ----------------------------------------------------------------- 6.1 pencere
def stage_windows(arch, seeds, max_epochs):
    hp = best_hp(arch)
    main = json.load(open(os.path.join(RESULTS, f"main_{arch}.json")))
    out = {}
    for W in (24, 72, 168):
        if W == 168 and "seeds" in main:
            rows = [r for r in main["seeds"] if r["seed"] in seeds]
            print(f"[{arch}] W=168: ana deneyden {len(rows)} seed yeniden kullanildi", flush=True)
        else:
            rows = T.run_seeds(arch, W, hp, seeds, max_epochs)
        out[str(W)] = {"params": count_params(make_model(arch, W)), "rows": rows}
    json.dump(out, open(os.path.join(RESULTS, f"windows_{arch}.json"), "w"), indent=1)


# ----------------------------------------------------------------- 6.2 gradyan
def input_grad_profile(model, X):
    model.eval()
    X = X.clone().requires_grad_(True)
    model(X).sum().backward()
    return X.grad.norm(dim=2).mean(0).numpy()      # (T,) -- t=T-1 en son adim


def stage_grad(n_samples=256):
    data = D.hourly_data(168)
    X, _ = next(D.iterate(data, data["test"][:n_samples], n_samples, False))
    out = {}
    for arch in ARCHS:
        D.set_seed(0)
        init = make_model(arch, 168)
        trained = make_model(arch, 168)
        trained.load_state_dict(torch.load(os.path.join(RESULTS, f"ckpt_{arch}_s0_W168.pt")))
        g0 = input_grad_profile(init, X)[::-1]       # lag 0 = son adim
        g1 = input_grad_profile(trained, X)[::-1]
        out[arch] = {"init": g0.tolist(), "trained": g1.tolist()}
        print(f"[grad] {arch}: lag0={g1[0]:.3e} lag24={g1[24]:.3e} lag72={g1[72]:.3e} lag167={g1[167]:.3e}", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "grad_norm.json"), "w"))


# ----------------------------------------------------------------- 6.3 latency
def stage_latency(warmup=50, runs=1000, blocks=3):
    torch.set_num_threads(1)
    x = torch.randn(1, 168, D.N_FEATURES)
    out = {}
    for arch in ARCHS:
        m = make_model(arch, 168).eval()
        allt, per_block = [], []
        with torch.inference_mode():
            for _ in range(warmup):
                m(x)
            for b in range(blocks):
                ts = []
                for _ in range(runs):
                    t0 = time.perf_counter()
                    m(x)
                    ts.append((time.perf_counter() - t0) * 1000.0)
                per_block.append({"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))})
                allt += ts
        out[arch] = {"median_ms": float(np.median(allt)), "p95_ms": float(np.percentile(allt, 95)), "blocks": per_block}
        print(f"[latency] {arch:12s} median={out[arch]['median_ms']:.3f} ms  p95={out[arch]['p95_ms']:.3f} ms  (blok medyanlari: "
              f"{[round(b['median_ms'], 3) for b in per_block]})", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "latency.json"), "w"), indent=1)


# ----------------------------------------------------------------- 7 transfer
EPOCHS_ADAPT = 60
N_DAYS = (2, 7, 30)


def adapt(model, tdata, ends, method, hp, seed, epochs=EPOCHS_ADAPT, bs=16):
    if method == "probe":
        for p in model.parameters():
            p.requires_grad = False
        last = model.head[2]
        for p in last.parameters():
            p.requires_grad = True
        params, lr = list(last.parameters()), hp["lr"]
    elif method == "finetune":
        params, lr = list(model.parameters()), hp["lr"] * 0.1
    else:
        params, lr = list(model.parameters()), hp["lr"]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=hp["wd"])
    crit = nn.HuberLoss(delta=1.0)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.eval() if method == "probe" else model.train()
        for X, y in D.iterate(tdata, ends, bs, True, rng):
            opt.zero_grad()
            crit(model(X), y).backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    return model


def stage_transfer(arch, seeds, pre_epochs=60):
    hp = best_hp(arch)
    src = D.source12_data()
    tdata = D.target12_data(src)
    rows, pre_rows = [], []
    y_te = tdata["y"][tdata["test"]]
    base = {}
    for n in N_DAYS:
        ends = tdata["train_ends"](n)
        ytr = tdata["y"][ends]
        p = tdata["persist"][tdata["test"]]
        base[str(n)] = {
            "mean_pred_mae": float(np.mean(np.abs(y_te - ytr.mean()))),
            "zero_pred_mae": float(np.mean(np.abs(y_te))),
            "persistence_mae": float(np.mean(np.abs(y_te - p))),
            "n_train_samples": int(len(ends)),
        }
    print(f"[transfer {arch}] kaynak train={len(src['train'])} val={len(src['val'])} | hedef test={len(tdata['test'])} | baseline={base}", flush=True)
    for s in seeds:
        D.set_seed(s)
        pre = make_model(arch, 14, hp["dropout"], grid="12h")
        val_mae, ep = T.fit(pre, src, hp, pre_epochs, s, patience=8, bs=32)
        zs_mae, _, _ = T.evaluate(pre, tdata, tdata["test"])
        pre_rows.append({"seed": s, "src_val_mae": val_mae, "epochs": ep, "zero_shot_target_mae": zs_mae})
        state = copy.deepcopy(pre.state_dict())
        print(f"  [{arch} seed={s}] kaynak val MAE={val_mae:.4f} (ep={ep}) | adaptasyonsuz hedef MAE={zs_mae:.4f}", flush=True)
        for n in N_DAYS:
            ends = tdata["train_ends"](n)
            for method in ("scratch", "probe", "finetune"):
                D.set_seed(s)
                m = make_model(arch, 14, hp["dropout"], grid="12h")
                if method != "scratch":
                    m.load_state_dict(state)
                adapt(m, tdata, ends, method, hp, s)
                mae, rmse, _ = T.evaluate(m, tdata, tdata["test"])
                rows.append({"n": n, "method": method, "seed": s, "test_mae": mae, "test_rmse": rmse})
            last = {r["method"]: round(r["test_mae"], 4) for r in rows[-3:]}
            print(f"    n={n:2d}g  {last}", flush=True)
    json.dump({"rows": rows, "pretrain": pre_rows, "baselines": base,
               "info": {"test_samples": int(len(tdata["test"])), "adapt_epochs": EPOCHS_ADAPT,
                        "hp": hp, "params": count_params(make_model(arch, 14, grid="12h"))}},
              open(os.path.join(RESULTS, f"transfer_{arch}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["windows", "grad", "latency", "transfer"])
    ap.add_argument("--arch")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    seeds = list(range(a.seeds))
    if a.stage == "windows":
        stage_windows(a.arch, seeds, a.epochs)
    elif a.stage == "grad":
        stage_grad()
    elif a.stage == "latency":
        stage_latency()
    else:
        stage_transfer(a.arch, seeds)


if __name__ == "__main__":
    main()
