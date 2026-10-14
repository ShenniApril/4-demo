"""Independent load and clean timing checks, performed after all builds finish."""
import json,subprocess,sys
from pathlib import Path

for threads in [1,4]:
    subprocess.run([sys.executable,'evaluate_saved.py','--index','experiments/IVF256_PQ128.faiss',
                    '--nprobe','64','--count','200','--threads',str(threads),
                    '--output',f'experiments/threads_{threads}.json'],check=True)
# The initial SQ8 PCA run overlapped the short exact-data audit. Repeat cleanly.
for probe in [16,64,256]:
    name=f'PCA256_IVF256_SQ8_np{probe}'
    repeat=Path(f'experiments/{name}_repeat.json')
    subprocess.run([sys.executable,'evaluate_saved.py','--index','experiments/PCA256_IVF256_SQ8.faiss',
                    '--nprobe',str(probe),'--output',str(repeat)],check=True)
    original=Path(f'experiments/{name}.json'); r=json.loads(original.read_text()); measured=json.loads(repeat.read_text())
    assert abs(r['recall_at_10']-measured['recall_at_10'])<1e-12
    for key in ['p50_ms','p95_ms','latencies_ms','per_query_recall','query_order']:
        r[key]=measured[key]
    r['mean_ms']=sum(measured['latencies_ms'])/len(measured['latencies_ms'])
    r['timing_source']=repeat.name
    original.write_text(json.dumps(r,indent=2))
