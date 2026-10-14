"""Dev-1M feasibility experiments; never uses original base at search time."""
import argparse, gc, hashlib, json, os, platform, struct, time
from pathlib import Path
import faiss
import numpy as np
import psutil
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = Path('experiments')
OUT.mkdir(exist_ok=True)

def vectors(name):
    with open(ROOT / name, 'rb') as f: n, d = struct.unpack('<II', f.read(8))
    assert (ROOT / name).stat().st_size == 8 + n*d*4
    return np.memmap(ROOT / name, mode='r', dtype='<f4', offset=8, shape=(n,d))

def truth():
    with open(ROOT / 'groundtruth.ibin', 'rb') as f:
        n,k = struct.unpack('<II',f.read(8))
        return np.fromfile(f,dtype='<u4',count=n*k).reshape(n,k)[:,:10]

def evaluate(index, q, gt, count, seed=2026):
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(q))[:count]
    for i in order[:20]: index.search(q[i:i+1],10)
    lats=[]; recalls=[]; ids=[]
    for i in order:
        t=time.perf_counter(); _, found=index.search(q[i:i+1],10)
        lats.append((time.perf_counter()-t)*1000)
        row=found[0]
        assert len(set(row.tolist()))==10 and np.all((row>=0)&(row<1000000))
        recalls.append(len(set(row.tolist()) & set(gt[i].tolist()))/10)
        ids.append(row.tolist())
    return dict(query_count=count, recall_at_10=float(np.mean(recalls)),
                p50_ms=float(np.percentile(lats,50)),p95_ms=float(np.percentile(lats,95)),
                mean_ms=float(np.mean(lats)),query_order=order.tolist(),latencies_ms=lats,
                per_query_recall=recalls,returned_ids=ids)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['initial','pq','baseline','fullpq'],default='initial')
    args=ap.parse_args()
    faiss.omp_set_num_threads(4)
    with threadpool_limits(limits=4):
        meta=dict(cpu='Intel Core Ultra 5 125H',platform=platform.platform(),machine=platform.machine(),
                  logical_cpus=os.cpu_count(),memory=psutil.virtual_memory()._asdict(),
                  faiss=faiss.__version__,numpy=np.__version__,compile_options=faiss.get_compile_options(),
                  train_threads=4,search_threads=1,threadpools=threadpool_info())
        (OUT/'environment.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        manifest=json.loads((ROOT/'manifest.json').read_text())
        checks={}
        for name,entry in manifest['files'].items():
            h=hashlib.sha256()
            with open(ROOT/name,'rb') as f:
                while block:=f.read(8*1024*1024): h.update(block)
            checks[name]=dict(bytes=(ROOT/name).stat().st_size,sha256=h.hexdigest(),
                              passed=h.hexdigest()==entry['sha256'] and (ROOT/name).stat().st_size==entry['bytes'])
        assert all(x['passed'] for x in checks.values())
        (OUT/'data_checks.json').write_text(json.dumps(checks,indent=2))
        print('Data validated',flush=True)
        base=vectors('base.fbin'); query=np.array(vectors('query.fbin')); gt=truth()
        rng=np.random.default_rng(42); train=np.array(base[rng.choice(len(base),32768,replace=False)])
        if args.mode=='initial':
            t=time.perf_counter(); pca=faiss.PCAMatrix(1024,256); pca.train(train)
            faiss.write_VectorTransform(pca,str(OUT/'pca256.transform'))
            eig=faiss.vector_to_array(pca.eigenvalues)
            (OUT/'pca.json').write_text(json.dumps(dict(train_rows=len(train),train_sec=time.perf_counter()-t,
                      retained_variance_256=float(eig[:256].sum()/eig.sum()),
                      variance_by_dim={str(d):float(eig[:d].sum()/eig.sum()) for d in [32,64,96,128,256,512,768,1024]}),indent=2))
        pca=faiss.read_VectorTransform(str(OUT/'pca256.transform'))
        if args.mode=='initial': specs=['PCA256,IVF256,SQ8','PCA256,IVF256,SQ4']
        elif args.mode=='pq': specs=['PCA256,IVF256,PQ32','PCA256,IVF256,PQ64']
        elif args.mode=='fullpq': specs=['IVF256,PQ64','IVF256,PQ128']
        else: specs=['IVF256,SQ8']
        for spec in specs:
            name=spec.replace(',','_'); path=OUT/(name+'.faiss')
            t=time.perf_counter()
            if spec.startswith('PCA'):
                core=faiss.index_factory(256,spec.split(',',1)[1]); index=faiss.IndexPreTransform(pca,core)
            else:
                core=faiss.index_factory(1024,spec); index=core
            core.cp.niter=12; core.cp.seed=42
            if hasattr(core,'pq'): core.pq.cp.niter=12; core.pq.cp.seed=42
            index.train(train)
            train_sec=time.perf_counter()-t
            for start in range(0,len(base),16384): index.add(np.array(base[start:start+16384]))
            build_sec=time.perf_counter()-t
            faiss.write_index(index,str(path)); size=path.stat().st_size
            print(name,'built',round(build_sec,2),'sec',size,'bytes',flush=True)
            # Search through a freshly loaded index; no base accesses in evaluation.
            del index,core; gc.collect()
            t=time.perf_counter(); index=faiss.read_index(str(path)); load_sec=time.perf_counter()-t
            core=faiss.extract_index_ivf(index)
            faiss.omp_set_num_threads(1)
            with threadpool_limits(limits=1):
                for nprobe in [16,64,256]:
                    core.nprobe=nprobe
                    result=dict(algorithm=spec,nprobe=nprobe,seed=42,train_rows=len(train),
                                train_sec=train_sec,build_sec=build_sec,load_sec=load_sec,
                                index_bytes=size,bytes_per_vector=size/len(base),
                                rss_before_search=psutil.Process().memory_info().rss,
                                index_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    result.update(evaluate(index,query,gt,len(query)))
                    result['rss_after_search']=psutil.Process().memory_info().rss
                    (OUT/(name+f'_np{nprobe}.json')).write_text(json.dumps(result,indent=2))
                    print(name,nprobe,'recall',result['recall_at_10'],'p95',result['p95_ms'],flush=True)
            del index,core; gc.collect(); faiss.omp_set_num_threads(4)

if __name__=='__main__': main()
