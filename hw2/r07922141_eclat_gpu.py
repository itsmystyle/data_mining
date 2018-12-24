import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import numpy as np
import itertools
import gc
from timeit import default_timer as timer
import sys
gc.enable()

THREAD_NUM = 128
BLOCK_NUM = 128

def gpusum(data, DATA_SIZE, BLOCK_NUM, THREAD_NUM):
    result = np.empty([BLOCK_NUM], dtype=np.uint32)

    mod = SourceModule("""
        __device__ void warpReduce(volatile unsigned int *sdata, int tid, int blockSize) {
            if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
            if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
            if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
            if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
            if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
            if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
        }

        __global__ void sum(unsigned short *g_idata, unsigned int *g_odata, int *DATA_SIZE) { 
            extern __shared__ unsigned int sdata[];
            
            const int tid = threadIdx.x;
            unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
            unsigned int tmp = 0;
            unsigned int gridSize = blockDim.x*2*gridDim.x;
            
            while(i < *DATA_SIZE){
                tmp += (g_idata[i] + g_idata[i + blockDim.x]); 
                i += gridSize;
            }
            sdata[tid] = tmp;
            __syncthreads();
            
            if (blockDim.x >= 512) { 
                if (tid < 256) { sdata[tid] += sdata[tid + 256]; } 
                __syncthreads(); 
            } 
            if (blockDim.x >= 256) { 
                if (tid < 128) { sdata[tid] += sdata[tid + 128]; } 
                __syncthreads(); 
            } 
            if (blockDim.x >= 128) { 
                if (tid < 64) { sdata[tid] += sdata[tid + 64]; } 
                __syncthreads(); 
            }
            
            if(tid < 32) warpReduce(sdata, tid, blockDim.x);
            
            if(tid == 0) g_odata[blockIdx.x] = sdata[0];
        }
    """)

    sum_ = mod.get_function('sum')
    sum_(drv.In(data), drv.Out(result), drv.In(DATA_SIZE), block=(THREAD_NUM, 1, 1), grid=(BLOCK_NUM, 1), shared=THREAD_NUM*4)

    return np.sum(result)

def eclat(current, candidates):
    if not candidates:
        return
    
    _C = []
    nxt_candidates = []
    current = frozenset(current)
    
    for i in candidates:
        data = np.logical_and(bitvector_dict[current], bitvector_dict[i], dtype=np.int8).astype(np.int8)
        sm = gpusum(data.astype(np.uint16), np.uint32(txs_len), BLOCK_NUM, THREAD_NUM)
        if sm >= min_support:
            _C.append([current | i, sm, i])
            bitvector_dict[current | i] = data
            nxt_candidates.append(i)
        del sm
        del data
    
    for nxt, freq, i in _C:
        result.append([nxt, freq])
        nxt_candidates.remove(i)
        eclat(nxt, nxt_candidates)
    
    del nxt_candidates
    del _C
    del current

def main(input_path, m_supp, output_path):

    with open(input_path, 'r') as fin:
        inp = fin.readlines()

    # toy_example = ['1 3 4', '2 3 5', '1 2 3 5', '2 5']
    # inp = toy_example

    global min_support
    min_support = len(inp) * m_supp
    global txs_len
    txs_len = len(inp)

    global bitvector_dict
    bitvector_dict = {}

    for idx, data in enumerate(inp):
        v = [frozenset([int(i)]) for i in data.rstrip().split()]
        for item in v:
            if item not in bitvector_dict:
                bitvector_dict[item] = np.array([0] * txs_len, dtype=np.int8)
            bitvector_dict[item][idx] = 1 

    global result
    result = []

    start = timer()
    print('Start Timer...')

    cand = set(bitvector_dict.keys())
    bitvector_dict[frozenset([])] = np.array([1] * txs_len)
    eclat(set([]), cand)

    end = timer()
    print('Time used: ', end - start)
    
    answer = []
    for st, freq in result:
        lst = sorted(st, key=lambda x: x)
        answer.append([lst, freq])
    
    result = sorted(answer, key=lambda x: (-len(x[0]), x[0], -x[1]))
    
    print('Answer length: ', len(result))
    
    with open(output_path, 'w') as fout:
        for key, value in result:
            fout.write(' '.join(map(str, key)) + ' (' + str(value) + ')\n')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Not Enought arguments.')
    else:
        inp = sys.argv[1]
        ms = float(sys.argv[2])
        out = sys.argv[3]
        main(inp, ms, out)
