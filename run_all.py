import os
import sys
import shutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Proje kok dizini
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import load_kaggle_or_synthetic_data
from models import build_models, count_parameters
from train import run_experiment_5_seeds
from analysis import (
    analyze_context_windows,
    analyze_lstm_gradient_norm,
    benchmark_latency,
    analyze_transfer_learning
)

def sync_to_desktop():
    desktop_target = r"C:\Users\serxan.m\Desktop\kovan_project"
    source = os.path.dirname(os.path.abspath(__file__))
    try:
        if os.path.exists(desktop_target):
            # Dosyalari kopyala
            for root, dirs, files in os.walk(source):
                rel_path = os.path.relpath(root, source)
                dest_dir = os.path.join(desktop_target, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                for f in files:
                    s_file = os.path.join(root, f)
                    d_file = os.path.join(dest_dir, f)
                    shutil.copy2(s_file, d_file)
            print(f"[Sync] Tum proje Desktop/kovan_project klasorune kopyalandi!")
    except Exception as e:
        print(f"[Sync Uyari] {e}")

def main():
    print("=" * 60)
    print("KOVON AGIRLIK DEGISIMI TAHMINI - MASTER CALISTIRICI")
    print("Ahmet Hoca Odev Tam Pipeline (Asama 0 - 8)")
    print("=" * 60)
    
    # 1. Veri Hazirligi (Asama 1)
    print("\n[1/5] Veri Seti Kontrol Ediliyor...")
    raw_df = load_kaggle_or_synthetic_data()
    
    # 2. Model Parametre Dogrulamasi (Asama 3 & 4)
    print("\n[2/5] Iso-Parametre Dogrulamasi Yapiliyor (~50k)...")
    models = build_models(in_features=8)
    for name, m in models.items():
        print(f"  {name:20}: {count_parameters(m):,} parametre")
        
    # 3. Adil Karsilastirma (Asama 5 - 5 Seed)
    print("\n[3/5] 5-Seed Egitim ve Guven Araligi Hesaplamasi...")
    summary_df, _, _ = run_experiment_5_seeds(epochs=10)
    
    # 4. Derinlemesine Analizler (Asama 6)
    print("\n[4/5] Analizler Yapiliyor (Pencere, Gradyan, Latency)...")
    analyze_context_windows(raw_df, windows=[24, 72, 168], epochs=6)
    analyze_lstm_gradient_norm(input_dim=8, window_size=168)
    benchmark_latency(in_dim=8, window_size=168, warmup=20, runs=300)
    
    # 5. Transfer Ogrenme (Asama 7)
    print("\n[5/5] Transfer Ogrenme Deneyleri (n = 2, 7, 30 Gun)...")
    analyze_transfer_learning(raw_df, target_hive='schwartau', days_list=[2, 7, 30])
    
    # Desktop'a kopyala
    sync_to_desktop()
    
    print("\n" + "=" * 60)
    print("BUTUN DENEYLER VE ANALIZLER BASARIYLA TAMAMLANDI!")
    print("Sonuclar ve grafikler results/ ve Desktop/kovan_project altina kaydedildi.")
    print("=" * 60)

if __name__ == "__main__":
    main()
