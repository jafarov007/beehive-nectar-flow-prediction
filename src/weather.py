import os
import json
import urllib.request
import numpy as np
import pandas as pd

HIVES = {
    # yaklasik koordinatlar (Wuerzburg: Universitaet Wuerzburg HOBOS, Bad Schwartau: Schleswig-Holstein)
    "wurzburg": {"lat": 49.79, "lon": 9.95},
    "schwartau": {"lat": 53.92, "lon": 10.70},
}
START, END = "2016-12-31", "2019-06-02"
VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]


def fetch_weather(name, cache_dir="data"):
    """Open-Meteo Historical Weather API (ucretsiz, anahtarsiz). Saatlik UTC veri, CSV cache."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"weather_{name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    c = HIVES[name]
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={c['lat']}&longitude={c['lon']}&start_date={START}&end_date={END}"
        f"&hourly={','.join(VARS)}&timezone=UTC"
    )
    print(f"[Hava] Open-Meteo'dan indiriliyor: {name}")
    with urllib.request.urlopen(url, timeout=120) as r:
        js = json.loads(r.read().decode("utf-8"))
    h = js["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"]),
        "out_temp": h["temperature_2m"],
        "out_hum": h["relative_humidity_2m"],
        "out_precip": h["precipitation"],
        "out_wind": h["wind_speed_10m"],
    })
    df.to_csv(path, index=False)
    print(f"[Hava] Kaydedildi: {path} ({len(df)} saat)")
    return df.set_index("timestamp")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for n in HIVES:
        w = fetch_weather(n)
        print(n, w.index.min(), w.index.max(), "NaN:", int(w.isna().sum().sum()))
