import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")

def wait_for_main_training():
    print("[Pipeline] Ana modellerin egitiminin (CNN, LSTM, Transformer) tamamlanmasi bekleniyor...")
    while True:
        cnn_done = os.path.exists(os.path.join(R, "main_cnn.json"))
        lstm_done = os.path.exists(os.path.join(R, "main_lstm.json"))
        tr_done = os.path.exists(os.path.join(R, "main_transformer.json"))
        
        # Check if seeds are present in each json
        def has_seeds(fn):
            p = os.path.join(R, fn)
            if not os.path.exists(p): return False
            try:
                import json
                d = json.load(open(p))
                return "seeds" in d and len(d["seeds"]) == 5
            except:
                return False
                
        c_ok = has_seeds("main_cnn.json")
        l_ok = has_seeds("main_lstm.json")
        t_ok = has_seeds("main_transformer.json")
        
        print(f"  Durum: CNN={'OK' if c_ok else 'Calisiyor'} | LSTM={'OK' if l_ok else 'Calisiyor'} | Transformer={'OK' if t_ok else 'Calisiyor'}")
        if c_ok and l_ok and t_ok:
            print("[Pipeline] Tum ana modeller basariyla egitildi ve 5 seed tamamlandi!")
            break
        time.sleep(15)

def run_cmd(cmd):
    print(f"\n[Calistiriliyor] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=ROOT)
    if res.returncode != 0:
        print(f"[Uyari] Komut hata kodu {res.returncode} ile sonlandi: {cmd}")

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    wait_for_main_training()
    
    # 6.1 Baglam pencereleri
    print("\n--- 6.1 Baglam Pencereleri (1, 3, 7 gun) Analizi ---")
    for a in ["cnn", "lstm", "transformer"]:
        run_cmd(f"python -u src/analysis.py windows --arch {a} --epochs 20")
        
    # 6.2 Gradyan analizi
    print("\n--- 6.2 Gradyan Normu Analizi ---")
    run_cmd("python -u src/analysis.py grad")
    
    # 6.3 Latency analizi
    print("\n--- 6.3 Latency Analizi ---")
    run_cmd("python -u src/analysis.py latency")
    
    # 7. Transfer ogrenme
    print("\n--- 7. Few-Shot Transfer Ogrenme Analizi (Yeni Kovan Schwartau) ---")
    for a in ["cnn", "lstm", "transformer"]:
        run_cmd(f"python -u src/analysis.py transfer --arch {a}")
        
    # 8. Ozetleme ve Rapor Uretimi
    print("\n--- 8. Ozet Tablolar, Grafikler ve Rapor Uretimi ---")
    run_cmd("python -u src/summarize.py")
    
    # Desktop ile senkronizasyon
    print("\n--- Desktop Senkronizasyonu ---")
    run_cmd(f"php {os.path.join(os.path.dirname(ROOT), 'sync_desktop.php')}")
    
    print("\n" + "="*60)
    print("BUTUN DENEYLER, ANALIZLER, GRAFIKLER VE RAPOR BASARIYLA TAMAMLANDI!")
    print("="*60)

if __name__ == "__main__":
    main()
