"""
Egitim: AYNI tuning butcesi, AYNI seed'ler, AYNI erken durdurma; test seti yalnizca final'de bir kez olculur.
Kullanim:  python src/train.py --arch cnn --stage all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D                                # noqa: E402
from models import make_model, count_params      # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = [0, 1, 2, 3, 4]


def evaluate(model, data, idx, bs=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in D.iterate(data, idx, bs, False):
            preds.append(model(X).numpy())
    p = np.concatenate(preds) * data["sd_y"] + data["mu_y"]
    t = data["y"][idx]
    err = p - t
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), p


def fit(model, data, hp, max_epochs, seed, patience=5, bs=64, verbose=False):
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    crit = nn.HuberLoss(delta=1.0)
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(max_epochs):
        model.train()
        for X, y in D.iterate(data, data["train"], bs, True, rng):
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_mae, _, _ = evaluate(model, data, data["val"])
        sched.step(val_mae)
        ep_run = ep + 1
        if verbose:
            print(f"    ep{ep + 1:02d} val_mae={val_mae:.4f}", flush=True)
        if val_mae < best - 1e-5:
            best, bad = val_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best, ep_run


def sample_trials(n, seed=1234):
    """Tum mimariler icin AYNI hiperparametre adaylari (ayni tuning butcesi)."""
    r = np.random.default_rng(seed)
    return [{"lr": float(10 ** r.uniform(np.log10(3e-4), np.log10(5e-3))),
             "dropout": float(r.uniform(0.0, 0.3)),
             "wd": float(10 ** r.uniform(-5, -2))} for _ in range(n)]


def run_seeds(arch, W, hp, seeds, max_epochs, save_ckpt=False, data=None):
    data = data or D.hourly_data(W)
    rows = []
    for s in seeds:
        D.set_seed(s)
        m = make_model(arch, W, hp["dropout"])
        t0 = time.time()
        val_mae, ep = fit(m, data, hp, max_epochs, s)
        test_mae, test_rmse, _ = evaluate(m, data, data["test"])
        rows.append({"seed": s, "val_mae": val_mae, "test_mae": test_mae, "test_rmse": test_rmse,
                     "epochs": ep, "sec": time.time() - t0})
        print(f"  [{arch} W={W} seed={s}] val={val_mae:.4f} test_MAE={test_mae:.4f} RMSE={test_rmse:.4f} ep={ep} ({rows[-1]['sec']:.0f}s)", flush=True)
        if save_ckpt:
            os.makedirs(RESULTS, exist_ok=True)
            torch.save(m.state_dict(), os.path.join(RESULTS, f"ckpt_{arch}_s{s}_W{W}.pt"))
    return rows


def baselines(data):
    te, tr = data["test"], data["train_full"]
    y = data["y"][te]
    out = {
        "mean_pred_mae": float(np.mean(np.abs(y - data["y"][tr].mean()))),
        "zero_pred_mae": float(np.mean(np.abs(y))),
    }
    p = data["persist"][te]
    ok = ~np.isnan(p)
    out["persistence_mae"] = float(np.mean(np.abs(y[ok] - p[ok])))
    out["persistence_rmse"] = float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))
    out["mean_pred_rmse"] = float(np.sqrt(np.mean((y - data["y"][tr].mean()) ** 2)))
    out["n_test"] = int(len(te))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--stage", default="all", choices=["tune", "final", "all"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--tune_epochs", type=int, default=12)
    ap.add_argument("--final_epochs", type=int, default=30)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"main_{a.arch}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    W = 168
    data = D.hourly_data(W)
    res["params"] = count_params(make_model(a.arch, W))

    if a.stage in ("tune", "all"):
        trials = sample_trials(a.trials)
        out = []
        for i, hp in enumerate(trials):
            D.set_seed(0)
            m = make_model(a.arch, W, hp["dropout"])
            t0 = time.time()
            val_mae, ep = fit(m, data, hp, a.tune_epochs, 0)
            out.append({**hp, "val_mae": val_mae, "epochs": ep, "sec": time.time() - t0})
            print(f"[{a.arch}] trial {i + 1}/{len(trials)} lr={hp['lr']:.2e} do={hp['dropout']:.2f} wd={hp['wd']:.1e} -> val_MAE={val_mae:.4f} ({out[-1]['sec']:.0f}s)", flush=True)
        best = min(out, key=lambda r: r["val_mae"])
        res["tuning"] = out
        res["best_hp"] = {k: best[k] for k in ("lr", "dropout", "wd")}
        res["tune_budget"] = {"trials": a.trials, "max_epochs": a.tune_epochs, "selection": "val MAE (seed 0)"}
        json.dump(res, open(path, "w"), indent=1)
        print(f"[{a.arch}] en iyi hp: {res['best_hp']}", flush=True)

    if a.stage in ("final", "all"):
        res["seeds"] = run_seeds(a.arch, W, res["best_hp"], SEEDS, a.final_epochs, save_ckpt=True, data=data)
        res["final_budget"] = {"max_epochs": a.final_epochs, "patience": 5}
        json.dump(res, open(path, "w"), indent=1)

    if a.arch == "cnn":
        json.dump(baselines(data), open(os.path.join(RESULTS, "baselines.json"), "w"), indent=1)
    print(f"[{a.arch}] bitti -> {path}", flush=True)


if __name__ == "__main__":
    main()
