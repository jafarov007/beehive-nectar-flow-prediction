"""
Convenience entrypoint running the full benchmark reproduction pipeline.
"""
import sys
import subprocess

if __name__ == "__main__":
    print("=" * 65)
    print("Executing full Beehive Nectar Flow benchmark pipeline...")
    print("Running: python run_pipeline_stages.py")
    print("=" * 65)
    cmd = [sys.executable, "run_pipeline_stages.py"] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
