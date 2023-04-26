#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np

def compute_rle(np.ndarray[np.uint8_t, ndim=1] mask):

    counts = []
    cdef int prev_val = 0
    cdef int cnt = 0

    cdef Py_ssize_t i
    for i in range(mask.shape[0]):
        if mask[i] == prev_val:
            cnt +=1
        else:
            counts.append(cnt)
            prev_val = mask[i]
            cnt = 1
    counts.append(cnt)

    return counts