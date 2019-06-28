import numpy as np
from numpy.linalg import norm
import pdb
from distances import sbd_dep_multi
from centroids import kShape_centroid

def multidim_kShape(ts,K,shift,init = 0, verbose = 1):
	iter_ = 100
	ndim = len(ts.shape)
	if ndim == 2:
		print('Warning: univariate time series input found, running kShape instead...')
		from kShape import kShape
		return kShape(ts,K,shift,init)
	n,m,d = ts.shape
	Dist = np.zeros((n,K))
	
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
		prev_mem = mem;
		#pdb.set_trace()
		for k in range(K):
			cent[k,:,:] = kShape_centroid(mem, ts, k, cent[k,:,:], shift)
			
		for i in range(n):
			for k in range(K):
				dist_,_,_ = sbd_dep_multi(cent[k,:,:],ts[i,:,:],shift)
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
			finalNorm = err
		
	return mem,Dist,cent,finalNorm
