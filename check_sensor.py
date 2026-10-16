import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np
from weather import fetch_weather

d = "C:/Users/serxan.m/Desktop/archive/"
for name in ["wurzburg", "schwartau"]:
    t = pd.read_csv(d + f"temperature_{name}.csv", parse_dates=["timestamp"]).dropna().set_index("timestamp")["temperature"].resample("1h").mean()
    h = pd.read_csv(d + f"humidity_{name}.csv", parse_dates=["timestamp"]).dropna().set_index("timestamp")["humidity"].resample("1h").mean()
    w = fetch_weather(name)
    j = pd.concat([t, w["out_temp"]], axis=1).dropna()
    print("====", name, "n=", len(j))
    for lag in range(-3, 4):
        s = w["out_temp"].shift(lag)
        jj = pd.concat([t, s], axis=1).dropna()
        print(f"  lag {lag:+d}h  corr(hive_temp, outside_T2m) = {jj.corr().iloc[0,1]:.4f}")
    print("  hive temp mean/min/max:", round(t.mean(), 2), round(t.min(), 2), round(t.max(), 2))
    print("  outside T mean:", round(w['out_temp'].mean(), 2))
    # daily amplitude as a proxy: ambient swings ~10C, in-hive brood nest stays ~34C
    daily_range = (t.resample("1D").max() - t.resample("1D").min()).median()
    print("  median daily temp range (hive sensor):", round(daily_range, 2))
    jh = pd.concat([h, w["out_hum"]], axis=1).dropna()
    print("  corr(hive_hum, outside_RH):", round(jh.corr().iloc[0, 1], 4), "| hum<=0 count:", int((h <= 0).sum()), "hum>100:", int((h > 100).sum()))
