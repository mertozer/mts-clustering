import pdb
import sys

from numpy.linalg import norm, eig, eigh
from scipy.stats import zscore
import numpy as np

from distances import dhat_shift_dep,sbd_dep_multi


def ksc_centroid(mem, ts, k, cent, shift):
	if len(ts.shape) == 2:
		n,m = ts.shape
		d = 1
	elif len(ts.shape) == 3:
		n,m,d = ts.shape
	else:
		print('Something Wrong in centroids.py ...')
		sys.exit()
	n_ = len(np.where(mem==k)[0])
	if d==1:
		a = np.zeros((n_,m,1))
	else:
		a = np.zeros((n_,m,d))
	
	ai = 0
	
	for i in np.where(mem==k)[0]:
		if not cent.any():
			opt_a = ts[i]
		else:
			_,_,opt_a = dhat_shift_dep(cent,ts[i],shift)
		
		a[ai,:,:] = opt_a
		ai += 1
	
	
	centroid = np.zeros((m,d))
	if a.shape[0] == 0:
		return centroid
	
	for d_i in range(d):
		a_di = a[:,:,d_i]
		b = a_di/np.tile(norm(a_di,axis=1),(m,1)).T
		M = np.matmul(b.T,b) - (n_+1)*np.eye(m)

		w,v = eigh(M)
		
		ksc_di = v[:,np.where(w == max(w))[0][0]]
		
		
		dist1 = norm(a_di[0,:,] - ksc_di.T)
		dist2 = norm(a_di[0,:,] - (-ksc_di.T))
		
		if dist1 > dist2:
			ksc_di = -ksc_di
		
		if np.sum(ksc_di) < 0:
			ksc_di = -ksc_di
		
		centroid[:,d_i] = ksc_di
	return centroid

def kShape_centroid(mem, ts, k, cent, shift):
	if len(ts.shape) == 2:
		n,m = ts.shape
		d = 1
	elif len(ts.shape) == 3:
		n,m,d = ts.shape
	else:
		print('Something Wrong in centroids.py ...')
		sys.exit()
	n_ = len(np.where(mem==k)[0])
	
	if d==1:
		a = np.zeros((n_,m,1))
	else:
		a = np.zeros((n_,m,d))
	
	ai = 0
	
	for i in np.where(mem==k)[0]:
		if not cent.any():
			opt_a = ts[i]
		else:
			_,_,opt_a = sbd_dep_multi(cent,ts[i],shift)
		
		a[ai,:,:] = opt_a
		ai += 1
	
	centroid = np.zeros((m,d))
	
	if a.shape[0] == 0:
		return centroid
	
	for d_i in range(d):
		#pdb.set_trace()
		ncolumns = a[:,:,d_i].shape[1]
		#pdb.set_trace()
		Y = zscore(a[:,:,d_i], axis=1, ddof = 1)
		Y = np.nan_to_num(Y)
		S = np.matmul(Y.T, Y)
		P = (np.eye(ncolumns) - 1.0 / ncolumns * np.ones((ncolumns,ncolumns)))
		M = np.matmul(np.matmul(P,S),P)
		if np.sum(M) == 0:
			centroid[:,d_i] = np.zeros((1, ts.shape[1]));
		
		
		w, v = eigh(M)
		centroid_di = v[:,np.where(w == max(w))[0][0]]
		
		finddistance1 = np.sqrt(np.sum((a[0,:,d_i] - centroid_di.T)**2))
		finddistance2 = np.sqrt(np.sum((a[0,:,d_i] - (-centroid_di.T))**2))
		#pdb.set_trace()
		if finddistance1 < finddistance2:
			centroid_di = centroid_di
		else:
			centroid_di = -centroid_di
		
		centroid_di = zscore(centroid_di,ddof=1)
		centroid_di = np.nan_to_num(centroid_di)
		centroid[:,d_i] = centroid_di
	
	return centroid

def vkShape_centroid(mem, ts, k, cent, shift):
	if len(ts.shape) == 2:
		n,m = ts.shape
		d = 1
	elif len(ts.shape) == 3:
		n,m,d = ts.shape
	else:
		print('Something Wrong in centroids.py ...')
		sys.exit()
	n_ = len(np.where(mem==k)[0])
	
	if d==1:
		a = np.zeros((n_,m,1))
	else:
		a = np.zeros((n_,m,d))
	
	ai = 0
	
	for i in np.where(mem==k)[0]:
		if not cent.any():
			opt_a = ts[i]
		else:
			_,_,opt_a = sbd_dep_multi(cent,ts[i],shift)
		
		a[ai,:,:] = opt_a
		ai += 1
	
	centroid = np.zeros((m,d))
	covar = np.zeros((d, d));
	
	if a.shape[0] == 0:
		return centroid,covar
	
	for d_i in range(d):
		#pdb.set_trace()
		ncolumns = a[:,:,d_i].shape[1]
		#pdb.set_trace()
		Y = zscore(a[:,:,d_i], axis=1, ddof = 1)
		Y = np.nan_to_num(Y)
		S = np.matmul(Y.T, Y)
		P = (np.eye(ncolumns) - 1.0 / ncolumns * np.ones((ncolumns,ncolumns)))
		M = np.matmul(np.matmul(P,S),P)
		if np.sum(M) == 0:
			centroid[:,d_i] = np.zeros((1, ts.shape[1]));
		
		
		w, v = eigh(M)
		centroid_di = v[:,np.where(w == max(w))[0][0]]
		
		finddistance1 = np.sqrt(np.sum((a[0,:,d_i] - centroid_di.T)**2))
		finddistance2 = np.sqrt(np.sum((a[0,:,d_i] - (-centroid_di.T))**2))
		#pdb.set_trace()
		if finddistance1 < finddistance2:
			centroid_di = centroid_di
		else:
			centroid_di = -centroid_di
		
		centroid_di = zscore(centroid_di,ddof=1)
		centroid_di = np.nan_to_num(centroid_di)
		centroid[:,d_i] = centroid_di
	
	sum_ = 0.0;
	for i in range(a.shape[0]):
		x_i = a[i]
		covar_i = (np.matmul(x_i.T,x_i))/m;
		sum_ = sum_ + covar_i;
	
	covar = sum_/a.shape[0];
	
	return centroid, covar
