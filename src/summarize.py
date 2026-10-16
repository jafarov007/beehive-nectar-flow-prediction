"""
Tum sonuc JSON'larindan tablo + grafik uretir ve report/template.md icindeki {{...}} yer tutucularini doldurur.
Hicbir sayi elle yazilmaz; hepsi results/*.json dosyalarindan gelir.
Kullanim:  python src/summarize.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
REP = os.path.join(ROOT, "report")
ARCHS = ["cnn", "lstm", "transformer"]
NAME = {"cnn": "Causal CNN", "lstm": "LSTM", "transformer": "Causal Transformer"}
COL = {"cnn": "#1f77b4", "lstm": "#d62728", "transformer": "#2ca02c"}
METH = {"scratch": "Sıfırdan", "probe": "Linear probe", "finetune": "Fine-tuning"}


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(h)


def fmt(m, h, d=4):
    return f"{m:.{d}f} ± {h:.{d}f}"


def table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


out = {}

# ------------------------------------------------------------------ ana karsilastirma
main = {a: load(f"main_{a}.json") for a in ARCHS}
base = load("baselines.json")
if all(main[a] and "seeds" in main[a] for a in ARCHS):
    rows, maes = [], {}
    for a in ARCHS:
        mae = [r["test_mae"] for r in main[a]["seeds"]]
        rmse = [r["test_rmse"] for r in main[a]["seeds"]]
        maes[a] = mae
        m1, h1 = ci(mae)
        m2, h2 = ci(rmse)
        rows.append([NAME[a], f"{main[a]['params']:,}", fmt(m1, h1), fmt(m2, h2),
                     f"{np.mean([r['epochs'] for r in main[a]['seeds']]):.1f}"])
    if base:
        rows.append(["*Baseline: eğitim ortalaması*", "-", f"{base['mean_pred_mae']:.4f}", f"{base['mean_pred_rmse']:.4f}", "-"])
        rows.append(["*Baseline: persistence (dünkü değişim)*", "-", f"{base['persistence_mae']:.4f}", f"{base['persistence_rmse']:.4f}", "-"])
        rows.append(["*Baseline: sıfır tahmin*", "-", f"{base['zero_pred_mae']:.4f}", "-", "-"])
    out["TABLE_MAIN"] = table(["Model", "Parametre", "Test MAE (kg/24s), ort ± %95 GA", "Test RMSE (kg/24s)", "Ort. epoch"], rows)
    # ikili Welch testleri
    prs = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = ARCHS[i], ARCHS[j]
            t, p = stats.ttest_ind(maes[a], maes[b], equal_var=False)
            ma, ha = ci(maes[a]); mb, hb = ci(maes[b])
            overlap = "evet" if (ma + ha >= mb - hb and mb + hb >= ma - ha) else "hayır"
            prs.append([f"{NAME[a]} vs {NAME[b]}", f"{ma - mb:+.4f}", f"{p:.4f}", overlap])
    out["TABLE_PAIRS"] = table(["Karşılaştırma", "ΔMAE (ilk − ikinci)", "Welch p", "GA'lar çakışıyor mu?"], prs)
    tun = []
    for a in ARCHS:
        b = main[a]["best_hp"]
        vs = [t["val_mae"] for t in main[a]["tuning"]]
        tun.append([NAME[a], len(vs), f"{min(vs):.4f}", f"{max(vs):.4f}", f"{b['lr']:.2e}", f"{b['dropout']:.2f}", f"{b['wd']:.1e}"])
    out["TABLE_TUNING"] = table(["Model", "Deneme", "En iyi val MAE", "En kötü val MAE", "lr", "dropout", "weight decay"], tun)

    fig, ax = plt.subplots(figsize=(7, 4))
    ms = [ci(maes[a]) for a in ARCHS]
    ax.bar([NAME[a] for a in ARCHS], [m for m, _ in ms], yerr=[h for _, h in ms], color=[COL[a] for a in ARCHS], capsize=6)
    if base:
        ax.axhline(base["mean_pred_mae"], ls="--", c="gray", label="eğitim ortalaması")
        ax.axhline(base["persistence_mae"], ls=":", c="k", label="persistence")
    ax.set_ylabel("Test MAE (kg / 24 saat)"); ax.set_title("Kilitli test: 5 seed, %95 güven aralığı"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig_main.png"), dpi=160); plt.close()

# ------------------------------------------------------------------ pencere
win = {a: load(f"windows_{a}.json") for a in ARCHS}
if all(win.values()):
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in ARCHS:
        xs, ys, es = [], [], []
        for W in ("24", "72", "168"):
            mae = [r["test_mae"] for r in win[a][W]["rows"]]
            m, h = ci(mae)
            xs.append(int(W) // 24); ys.append(m); es.append(h)
            rows.append([NAME[a], f"{int(W) // 24} gün ({W} s)", f"{win[a][W]['params']:,}", fmt(m, h), len(mae)])
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=NAME[a], color=COL[a])
    ax.set_xticks([1, 3, 7]); ax.set_xlabel("Girdi penceresi (gün)"); ax.set_ylabel("Test MAE (kg / 24 saat)")
    ax.set_title("Bağlam uzunluğuna tepki"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig_window.png"), dpi=160); plt.close()
    out["TABLE_WINDOW"] = table(["Model", "Pencere", "Parametre", "Test MAE ± %95 GA", "#seed"], rows)

# ------------------------------------------------------------------ gradyan
g = load("grad_norm.json")
if g:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    lag = np.arange(168)
    axs[0].semilogy(lag, g["lstm"]["trained"], c=COL["lstm"], label="LSTM (eğitilmiş)")
    axs[0].semilogy(lag, g["lstm"]["init"], c=COL["lstm"], ls="--", label="LSTM (rastgele başlangıç)")
    axs[0].set_title("LSTM: ‖∂ŷ/∂x_t‖ (zaman geriye doğru)")
    for a in ARCHS:
        axs[1].semilogy(lag, g[a]["trained"], c=COL[a], label=NAME[a])
    axs[1].set_title("Eğitilmiş üç model")
    for ax in axs:
        ax.set_xlabel("Gecikme (saat önce; 0 = tahmin anı)"); ax.set_ylabel("Ortalama gradyan normu (log)")
        ax.grid(alpha=.3, which="both"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig_grad.png"), dpi=160); plt.close()
    rows = []
    for a in ARCHS:
        t = g[a]["trained"]
        rows.append([NAME[a]] + [f"{t[k]:.2e}" for k in (0, 6, 24, 72, 167)] + [f"{t[167] / t[0]:.2e}"])
    out["TABLE_GRAD"] = table(["Model (eğitilmiş)", "lag 0", "lag 6", "lag 24", "lag 72", "lag 167", "lag167 / lag0"], rows)

# ------------------------------------------------------------------ latency
lat = load("latency.json")
if lat:
    rows = [[NAME[a], f"{lat[a]['median_ms']:.3f}", f"{lat[a]['p95_ms']:.3f}",
             ", ".join(f"{b['median_ms']:.3f}" for b in lat[a]["blocks"])] for a in ARCHS]
    out["TABLE_LATENCY"] = table(["Model", "Median (ms)", "p95 (ms)", "3 blok medyanı (ms)"], rows)

# ------------------------------------------------------------------ transfer
tr = {a: load(f"transfer_{a}.json") for a in ARCHS}
if all(tr.values()):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    rows = []
    for ax, a in zip(axs, ARCHS):
        for mth, c in zip(("scratch", "probe", "finetune"), ("#7f7f7f", "#ff7f0e", "#1f77b4")):
            xs, ys, es = [], [], []
            for n in (2, 7, 30):
                mae = [r["test_mae"] for r in tr[a]["rows"] if r["n"] == n and r["method"] == mth]
                m, h = ci(mae)
                xs.append(n); ys.append(m); es.append(h)
                rows.append([NAME[a], f"{n} gün", METH[mth], fmt(m, h), len(mae)])
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=METH[mth], color=c)
        zs = ci([p["zero_shot_target_mae"] for p in tr[a]["pretrain"]])[0]
        ax.axhline(zs, c="g", ls="-.", label="Önceden eğitilmiş (adaptasyonsuz)")
        b = tr[a]["baselines"]
        ax.plot([2, 7, 30], [b[str(n)]["mean_pred_mae"] for n in (2, 7, 30)], "k:", label="n-gün ortalama baseline")
        ax.axhline(b["30"]["zero_pred_mae"], c="m", ls=":", label="sıfır tahmin")
        ax.set_xscale("log"); ax.set_xticks([2, 7, 30]); ax.set_xticklabels(["2", "7", "30"])
        ax.set_title(NAME[a]); ax.set_xlabel("Yeni kovandan etiketli veri (gün)"); ax.grid(alpha=.3)
    axs[0].set_ylabel("Test MAE (kg / 24 saat)"); axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig_transfer.png"), dpi=160); plt.close()
    out["TABLE_TRANSFER"] = table(["Model", "n", "Yöntem", "Test MAE ± %95 GA", "#seed"], rows)
    brow = []
    b = tr["cnn"]["baselines"]
    for n in (2, 7, 30):
        brow.append([f"{n} gün", b[str(n)]["n_train_samples"], f"{b[str(n)]['mean_pred_mae']:.4f}",
                     f"{b[str(n)]['zero_pred_mae']:.4f}", f"{b[str(n)]['persistence_mae']:.4f}"])
    out["TABLE_TRANSFER_BASE"] = table(["n", "#eğitim örneği", "n-gün ortalaması", "sıfır tahmin", "persistence"], brow)
    zrow = [[NAME[a], fmt(*ci([p["src_val_mae"] for p in tr[a]["pretrain"]])), fmt(*ci([p["zero_shot_target_mae"] for p in tr[a]["pretrain"]]))] for a in ARCHS]
    out["TABLE_TRANSFER_PRE"] = table(["Model", "Kaynak (Würzburg 12s) val MAE", "Yeni kovanda adaptasyonsuz MAE"], zrow)

# ------------------------------------------------------------------ yaz
os.makedirs(REP, exist_ok=True)
json.dump({k: v for k, v in out.items()}, open(os.path.join(R, "tables.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tpl = os.path.join(REP, "template.md")
if os.path.exists(tpl):
    txt = open(tpl, encoding="utf-8").read()
    for k, v in out.items():
        txt = txt.replace("{{" + k + "}}", v)
    open(os.path.join(REP, "sonuc_ve_rapor.md"), "w", encoding="utf-8").write(txt)
    left = [w for w in __import__("re").findall(r"\{\{(\w+)\}\}", txt)]
    print("rapor yazildi; doldurulamayan yer tutucular:", left)
sys.stdout.reconfigure(encoding="utf-8")
for k, v in out.items():
    print("\n##", k, "\n" + v)
