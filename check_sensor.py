import os
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import data as D

print("=" * 60)
print("SENSOR VALIDATION: Hive Telemetry & Meteorology Alignment")
print("=" * 60)

for name in ["wurzburg", "schwartau"]:
    if name == "wurzburg":
        df, _ = D.build_hourly(name)
    else:
        df, _ = D.build_12h(name)
    print(f"\n==== Hive: {name.capitalize()} (gözlem sayısı n={len(df):,}) ====")
    print(f"  Kovan İçi Sıcaklık (ort / min / max) : {df['t_in'].mean():.2f}°C / {df['t_in'].min():.2f}°C / {df['t_in'].max():.2f}°C")
    print(f"  Dış Ortam Sıcaklığı (ort)           : {df['t_out'].mean():.2f}°C")
    print(f"  Korelasyon (İç Sıcaklık ~ Dış Sıcaklık): {df['t_in'].corr(df['t_out']):.4f}")
    print(f"  Korelasyon (İç Nem ~ Dış Nem)          : {df['h_in'].corr(df['h_out']):.4f}")
    daily_range = (df["t_in"].resample("1D").max() - df["t_in"].resample("1D").min()).median()
    print(f"  Medyan Günlük Sıcaklık Salınımı      : {daily_range:.2f}°C")
