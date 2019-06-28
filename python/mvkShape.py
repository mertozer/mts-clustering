import numpy as np
from numpy.linalg import norm
import pdb
from distances import sbd_dep_multi
from centroids import vkShape_centroid
from joblib import Parallel
import time

def multidim_vkShape(ts,K,shift,init = 0, verbose = 1, alpha=0):
	iter_ = 100
	ndim = len(ts.shape)
	if ndim == 2:
		print('Warning: univariate time series input found, running kShape instead...')
		from kShape import kShape
		return kShape(ts,K,shift,init)
	n,m,d = ts.shape
	Dist = np.zeros((n,K))
	covar = np.zeros((K, d, d));
	
	#np.random.seed(42)
	if init == 0:
		mem = np.ceil(K*np.random.rand(n)) - 1
		#mem = np.zeros((n))
		#ii = 0
		#with open('compareMatlab/memInit') as fi:
		#	for line in fi:
		#		mem[ii] = int(line.rstrip('\n'))-1
		#		ii += 1
		cent = np.zeros((K,m,d))
	else:
		cent = init
		for i in range(n):
			for k in range(K):
				Dist[i,k],_,_ = sbd_dep_multi(cent[k,:,:],ts[i,:,:],shift)
		mem = np.argmin(Dist,axis=1)

	prevErr = -1
	try_ = 0
	for it in range(iter_):
		if verbose:
			print('Iteration',it)
			start_time = time.time()
		prev_mem = mem;
		#pdb.set_trace()
		with Parallel(n_jobs=15) as parallel: 
			for k in range(K):
				cent[k,:,:],covar[k,:,:] = vkShape_centroid(mem, ts, k, cent[k,:,:], shift)
		with Parallel(n_jobs=15) as parallel: 
			for i in range(n):
				x_i = ts[i];
				
				covar_i = np.matmul(x_i.T,x_i)/m;
				for k in range(K):
					dist_,_,_ = sbd_dep_multi(cent[k,:,:],ts[i,:,:],shift)
					dist_ += alpha*norm(covar[k,:,:]-covar_i,'fro')**2;
				   
					Dist[i,k] = dist_
		
		mem = np.argmin(Dist,axis=1)
		
		err = norm(prev_mem-mem)
		
		if err == 0:
			break
		else:
			if err == prevErr:
				try_ = try_ + 1
				if try_ > 2:
					break
			else:
				prevErr = err
				try_ = 0
		if verbose:
			print('||PrevMem-CurMem||=',err)
			print(time.time()-start_time,'secs')
		finalNorm = err
		
	return mem,Dist,cent,covar,finalNorm
