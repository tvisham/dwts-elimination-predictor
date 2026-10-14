"""Weekly refresh: scrape latest scores -> pull views -> retrain -> simulate -> web/data.js."""
import subprocess
import sys

for step in ["scrape.py", "popularity.py", "photos.py", "model.py"]:
    print(f"== {step}", flush=True)
    subprocess.run([sys.executable, step], check=True)
print("done: open web/index.html via  python -m http.server 8765 --directory web")
