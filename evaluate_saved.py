"""Independent saved-index evaluator: deliberately never opens base.fbin."""
import argparse,hashlib,json,struct,time
from pathlib import Path
import faiss
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--index',required=True)
    ap.add_argument('--nprobe',type=int,default=64); ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--count',type=int,default=1000); ap.add_argument('--output',required=True)
    a=ap.parse_args(); faiss.omp_set_num_threads(a.threads)
    with open('query.fbin','rb') as f:
        n,d=struct.unpack('<II',f.read(8)); q=np.fromfile(f,dtype='<f4').reshape(n,d)
    with open('groundtruth.ibin','rb') as f:
        n,k=struct.unpack('<II',f.read(8)); gt=np.fromfile(f,dtype='<u4',count=n*k).reshape(n,k)[:,:10]
    rss=psutil.Process().memory_info().rss; t=time.perf_counter()
    index=faiss.read_index(a.index); load=time.perf_counter()-t
    delta=psutil.Process().memory_info().rss-rss; faiss.extract_index_ivf(index).nprobe=a.nprobe
    order=np.random.default_rng(2026).permutation(len(q))[:a.count]; lats=[]; recalls=[]
    with threadpool_limits(limits=a.threads):
        for i in order[:20]: index.search(q[i:i+1],10)
        for i in order:
            t=time.perf_counter(); _,ids=index.search(q[i:i+1],10); lats.append((time.perf_counter()-t)*1000)
            assert len(set(ids[0]))==10 and np.all((ids[0]>=0)&(ids[0]<1000000))
            recalls.append(len(set(ids[0]) & set(gt[i]))/10)
    result=dict(index=a.index,nprobe=a.nprobe,threads=a.threads,count=a.count,
                recall_at_10=float(np.mean(recalls)),p50_ms=float(np.percentile(lats,50)),p95_ms=float(np.percentile(lats,95)),
                index_bytes=Path(a.index).stat().st_size,load_sec=load,load_rss_delta_bytes=delta,
                base_opened=False,latencies_ms=lats,per_query_recall=recalls,query_order=order.tolist())
    Path(a.output).write_text(json.dumps(result,indent=2)); print({k:v for k,v in result.items() if not isinstance(v,list)},flush=True)

if __name__=='__main__': main()
