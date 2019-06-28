import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import comb

def rand_index(yhat, y):
    tp_plus_fp = comb(np.bincount(yhat), 2).sum()
    tp_plus_fn = comb(np.bincount(y), 2).sum()
    A = np.c_[(yhat, y)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(yhat))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def toy_dataset(type_):
	toy_dataset_single = np.array(
		[
			[1,2,3,4,5],
			[2,3,4,5,6],
			[3,4,5,6,7],
			[7,6,5,4,3],
			[6,5,4,3,2],
			[5,4,3,2,1]
		]
	)
	toy_dataset = np.array(
		[
			[
				[1,5],
				[2,4],
				[3,3],
				[4,2],
				[5,1]
			],
			[
				[2,6],
				[3,5],
				[4,4],
				[5,3],
				[6,2]
			],
			[
				[3,7],
				[4,6],
				[5,5],
				[6,4],
				[7,3]
			],
			[
				[5,1],
				[4,2],
				[3,3],
				[2,4],
				[1,5]
			],
			[
				[6,2],
				[5,3],
				[4,4],
				[3,5],
				[2,6]
			],
			[
				[7,3],
				[6,4],
				[5,5],
				[4,6],
				[3,7]
			]
		]
	)
	if type_ == 'single':
		return toy_dataset_single
	elif type_ == 'multi':
		return toy_dataset
	else:
		print('Warning choose input parameter as single or multi...')
