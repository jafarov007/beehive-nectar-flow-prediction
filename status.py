import os
import sys
import json
import time

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")
LOG_DIR = r"C:\Users\serxan.m\.gemini\antigravity-ide\brain\a388d38c-be7d-4252-9702-9566d1cfe7c9\.system_generated\tasks"

def get_last_lines(filepath, n=3):
    if not os.path.exists(filepath):
        return ["Henüz başlamadı."]
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
            return lines[-n:] if lines else ["Çalışıyor (çıktı bekleniyor)..."]
    except Exception as e:
        return [f"Okunamadı: {e}"]

def check_json(filename):
    p = os.path.join(R, filename)
    if not os.path.exists(p):
        return "❌ Bekliyor"
    try:
        d = json.load(open(p, "r", encoding="utf-8"))
        if "seeds" in d and len(d["seeds"]) == 5:
            return "✅ 5 Seed Tamamlandı"
        elif "tuning" in d:
            return "⏳ Tuning Bitti, Seedler Eğitiliyor"
        return "⏳ Devam ediyor"
    except:
        return "⏳ Yazılıyor..."

def main():
    print("=" * 65)
    print("      KOVAN PROJESİ — CANLI DURUM VE AŞAMA TAKİP PANELİ")
    print("=" * 65)
    print(f"Zaman: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Ana Modeller
    print("[1] ANA MODELLER (Aşama 4 & 5 - 5 Seed):")
    for arch, fn, log_id in [("Causal CNN", "main_cnn.json", "task-277.log"),
                             ("LSTM", "main_lstm.json", "task-278.log"),
                             ("Causal Transformer", "main_transformer.json", "task-279.log")]:
        st = check_json(fn)
        log_path = os.path.join(LOG_DIR, log_id)
        last_log = get_last_lines(log_path, 1)[0]
        print(f"  * {arch:20s}: {st}")
        print(f"    Son Durum: {last_log}")
        
    # 2. Analiz Aşamaları
    print("\n[2] ANALİZ AŞAMALARI (Aşama 6):")
    w_ok = os.path.exists(os.path.join(R, "windows_cnn.json"))
    g_ok = os.path.exists(os.path.join(R, "grad_norm.json"))
    l_ok = os.path.exists(os.path.join(R, "latency.json"))
    print(f"  * Girdi Pencereleri (1/3/7 gün) : {'✅ Tamamlandı' if w_ok else '⏳ Bekliyor'}")
    print(f"  * LSTM Gradyan Sönümlenmesi     : {'✅ Tamamlandı' if g_ok else '⏳ Bekliyor'}")
    print(f"  * Latency (Çıkarım Gecikmesi)  : {'✅ Tamamlandı' if l_ok else '⏳ Bekliyor'}")
    
    # 3. Transfer Öğrenme
    print("\n[3] FEW-SHOT TRANSFER (Aşama 7 - Bad Schwartau):")
    t_cnn = os.path.exists(os.path.join(R, "transfer_cnn.json"))
    print(f"  * Yeni Kovan Transferi         : {'✅ Tamamlandı' if t_cnn else '⏳ Bekliyor'}")
    
    # 4. Rapor ve Tablolar
    print("\n[4] YEKUN RAPOR VE GRAFİKLER (Aşama 8):")
    rep_ok = os.path.exists(os.path.join(ROOT, "report", "sonuc_ve_rapor.md"))
    fig_ok = os.path.exists(os.path.join(R, "fig_main.png"))
    print(f"  * Grafikler (.png)             : {'✅ Hazır' if fig_ok else '⏳ Bekliyor'}")
    print(f"  * Final Raporu (sonuc_ve_rapor): {'✅ Hazır' if rep_ok else '⏳ Bekliyor'}")
    
    print("\n" + "=" * 65)
    print("İpucu: İstediğiniz an 'python status.py' yazarak son durumu görebilirsiniz.")
    print("=" * 65)

if __name__ == "__main__":
    main()
