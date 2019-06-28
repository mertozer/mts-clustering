import numpy as np
from numpy.linalg import norm
import pdb
from distances import dhat_shift_dep
from centroids import ksc_centroid

def multidim_kSC(ts,K,shift,init = 0, verbose = 1):
	iter_ = 100
	ndim = len(ts.shape)
	if ndim == 2:
		print('Warning: univariate time series input found, running kSC instead...')
		from ksc import kSC
		return kSC(ts,K,shift,init)
	n,m,d = ts.shape
	Dist = np.zeros((n,K))
	
	np.random.seed(42)
	if init == 0:
		mem = np.ceil(K*np.random.rand(n,1))-1
		cent = np.zeros((K,m,d))
	else:
		cent = init
		for i in range(n):
			for k in range(K):
				Dist[i,k],_,_ = dhat_shift_dep(cent[k,:,:],ts[i,:,:],shift)
		mem = np.argmin(Dist,axis=1)
	
	prevErr = -1
	try_ = 0
	
	for it in range(iter_):
		if verbose:
			print('Iteration',it)
		prev_mem = mem;
		for k in range(K):
			cent[k,:,:] = ksc_centroid(mem, ts, k, cent[k,:,:], shift)
		
		for i in range(n):
			for k in range(K):
				Dist[i,k],_,_ = dhat_shift_dep(cent[k,:,:],ts[i,:,:],shift)
		
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
		
	finalNorm = err
	return mem,Dist,cent,finalNorm
