import numpy as np
from numpy.linalg import norm
from numpy.fft import fft, ifft
import pdb
import sys

def dhat_shift_dep(t1, t2, shift):
	sanity_check(t1,t2)
	
	d = t1.shape[1]
	
	min_d = 0
	for d_i in range(d):
		min_d += scale_d(t1[:,d_i],t2[:,d_i])
	opt_t2 = t2
	optShift = 0
	for sh in range(-shift,shift+1):
		if sh == 0:
			continue
		elif sh < 0:
			shifted_t2 = np.append(t2[-sh:len(t2),:],np.zeros((-sh,d)),axis=0)
		else:
			shifted_t2 = np.append(np.zeros((sh,d)),t2[0:len(t2)-sh,:],axis=0)
			
		cur_d = 0
		for d_i in range(d):
			cur_d += scale_d(t1[:,d_i],shifted_t2[:,d_i])
		
		if cur_d <= min_d:
			optShift = sh
			opt_t2 = shifted_t2
			min_d = cur_d
	
	optShift = np.ones((1,d))*optShift
	dist = min_d
	return dist, optShift, opt_t2


def scale_d(t1,t2):
	alpha = np.matmul(t1,t2.T)/(np.matmul(t2,t2.T)+np.finfo(float).eps)
	dist = norm(t1 - t2*alpha)/(norm(t1)+np.finfo(float).eps)
	return dist


	
def sbd_dep_multi(t1,t2,shift):
	sanity_check(t1,t2)
	
	d = t1.shape[1]
	
	cc_ = 0
	for d_i in range(d):
		cc_ += NCCc(t1[:,d_i],t2[:,d_i])
	
	'''cc = np.zeros((cc_.shape))
	cc[t1.shape[0]] = cc_[t1.shape[0]]
	
	for i in range(shift):
		cc[t1.shape[0] + i] = cc_[t1.shape[0] + i]
		cc[t1.shape[0] - i] = cc_[t1.shape[0] - i]
	'''
	maxCC = np.max(cc_)
	maxCCI = np.argmax(cc_)
	
	sh = maxCCI - max(t1.shape[0]-1,t2.shape[0]-1)
	
	if sh < 0:
		shifted_t2 = np.append(t2[-sh:len(t2),:],np.zeros((-sh,d)),axis=0)
	else:
		shifted_t2 = np.append(np.zeros((sh,d)),t2[0:len(t2)-sh,:],axis=0)
	
	optShift = np.ones((1,d))*sh
	opt_t2 = shifted_t2
	dist = d - maxCC
	
	return dist, optShift, opt_t2
	
def NCCc(t1,t2):
	len_ = len(t1)
	fftLen = int(2**np.ceil(np.log2(abs(2*len_ - 1))))
	
	r = ifft(fft(t1, fftLen) * np.conj(fft(t2, fftLen)))
	r = np.concatenate((r[-(len_-1):], r[:len_]))
	
	return np.real(r)/((norm(t1) * norm(t2)) + np.finfo(float).eps)

def sanity_check(t1,t2):
	## sanity checks
	if len(t1.shape) == 2 and len(t2.shape) == 2:
		if not t1.shape[0] == t2.shape[0] or not t1.shape[1] == t2.shape[1]:
			print('Sth wrong with your time series shapes:',t1.shape,t2.shape)
			sys.exit()
	elif len(t1.shape) == 1 and len(t2.shape) == 1:
		if not t1.shape[0] == t2.shape[0]:
			print('Sth wrong with your time series shapes:',t1.shape,t2.shape)
			sys.exit()
	else:
		print('Sth wrong with your time series shapes:',t1.shape,t2.shape)
		sys.exit()
