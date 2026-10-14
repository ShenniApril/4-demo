"""Sequential runs avoid contaminating single-query latency with other builds."""
import subprocess,sys

for mode in ['fullpq','baseline']:
    subprocess.run([sys.executable,'benchmark.py','--mode',mode],check=True)
subprocess.run([sys.executable,'evaluate_saved.py','--index','experiments/IVF256_PQ128.faiss',
                '--nprobe','64','--output','experiments/independent_load.json'],check=True)
for threads in [1,4]:
    subprocess.run([sys.executable,'evaluate_saved.py','--index','experiments/IVF256_PQ128.faiss',
                    '--nprobe','64','--count','200','--threads',str(threads),
                    '--output',f'experiments/threads_{threads}.json'],check=True)
