import numpy as np
from pylog import *

@pylog
def GEMM(m1_int, m2_int, prod_int):
    m1 = np.empty((128, 128), np.int)
    m2 = np.empty((128, 128), np.int)
    prod = np.empty((128, 128), np.int)
    pragma("HLS array_partition variable=m1 cyclic factor=8 dim=0")
    pragma("HLS array_partition variable=m2 cyclic factor=8 dim=0")
    pragma("HLS array_partition variable=prod cyclic factor=6 dim=0")
    for i in range(128):
        for j in range(128):
            m1[i][j] = m1_int[i][j]
            m2[i][j] = m2_int[i][j]
            prod[i][j] = 0
    for jj in range(16):
        for kk in range(16):
            for i in range(128).unroll(8):
                for j in range (8).unroll(8):
                    for k in range(8).unroll(8):
                        prod[i][8 * jj + j] = m1[i][8 * kk + k] * m2[8 * kk + k][8 * jj + j];

    for i in range(128):
        for j in range(128):
            prod_int[i][j] = prod[i][j]

if __name__ == "__main__":
    m1_int = np.zeros((128, 128), np.int32)
    m2_int = np.zeros((128, 128), np.int32)
    prod_int = np.zeros((128, 128), np.int32)
    GEMM(m1_int, m2_int, prod_int)
    print(prod_int)
