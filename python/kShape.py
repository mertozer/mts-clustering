import numpy as np
from numpy.linalg import norm
import pdb
from distances import sbd_dep_multi
from centroids import kShape_centroid

def kShape(ts,K,shift,init = 0):
	iter_ = 100
	ndim = len(ts.shape)
	if ndim > 2:
		print('Warning: multivariate time series fed, running m-kShape instead...')
		from mkShape import multidim_kShape
		return multidim_kShape(ts,K,shift,init)
	n,m = ts.shape
	ts = np.reshape(ts,(ts.shape[0],ts.shape[1],1))
	Dist = np.zeros((n,K))
	
	
	if init == 0:
		mem = np.ceil(K*np.random.rand(n,1))-1
		cent = np.zeros((K,m,1))
	else:
		cent = init
		for i in range(n):
			for k in range(K):
				Dist[i,k],_,_ = sbd_dep_multi(cent[k,:],ts[i,:],shift)
		mem = np.argmin(Dist,axis=1)
	
	prevErr = -1
	try_ = 0
	
	for it in range(iter_):
		print('Iteration',it)
		prev_mem = mem;
		for k in range(K):
			cent[k] = kShape_centroid(mem, ts, k, cent[k,:], shift)
		
		for i in range(n):
			for k in range(K):
				Dist[i,k],_,_ = sbd_dep_multi(cent[k,:],ts[i,:],shift)
		
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
		print('||PrevMem-CurMem||=',err)
		
	finalNorm = err
	return mem,Dist,cent,finalNorm
