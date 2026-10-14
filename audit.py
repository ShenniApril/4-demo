import json,time
import numpy as np
import faiss
from threadpoolctl import threadpool_limits
from benchmark import vectors,truth,OUT,evaluate

base=vectors('base.fbin'); q=np.array(vectors('query.fbin')); gt=truth()
faiss.omp_set_num_threads(4)
with threadpool_limits(limits=4):
    t=time.perf_counter(); best_d=np.full((100,10),np.inf,dtype='float32'); best_i=np.zeros((100,10),dtype='int64')
    for start in range(0,len(base),16384):
        x=np.array(base[start:start+16384]); d=(q[:100]**2).sum(1)[:,None]+(x**2).sum(1)[None,:]-2*q[:100]@x.T
        loc=np.argpartition(d,9,axis=1)[:,:10]; val=np.take_along_axis(d,loc,axis=1)
        all_d=np.concatenate([best_d,val],axis=1); all_i=np.concatenate([best_i,loc+start],axis=1)
        pick=np.argsort(all_d,axis=1)[:,:10]; best_d=np.take_along_axis(all_d,pick,axis=1); best_i=np.take_along_axis(all_i,pick,axis=1)
    result=dict(exact_queries=100,recall=float(np.mean([len(set(i)&set(g))/10 for i,g in zip(best_i,gt[:100])])),sec=time.perf_counter()-t,
                method='blocked FP32 squared L2, top10; base only used in this validation stage')
    (OUT/'exact_audit.json').write_text(json.dumps(result,indent=2)); print(result,flush=True)
