import numpy as np
import itertools
import gc
from timeit import default_timer as timer
import sys
gc.enable()

def eclat(current, candidates):
    if not candidates:
        return
    
    _C = []
    nxt_candidates = []
    current = frozenset(current)
    
    for i in candidates:
        all_sum = np.sum(bitvector_dict[current] & bitvector_dict[i])
        sm = np.sum(np.logical_and(bitvector_dict[current], bitvector_dict[i]))
        if np.sum(np.logical_and(bitvector_dict[current], bitvector_dict[i])) >= min_support:
            tmp = np.sum(np.logical_and(bitvector_dict[current], bitvector_dict[i]))
            _C.append([current | i, np.sum(np.logical_and(bitvector_dict[current], bitvector_dict[i])), i])
            bitvector_dict[current | i] = np.logical_and(bitvector_dict[current], bitvector_dict[i])
            nxt_candidates.append(i)
            del tmp
        del sm
        del all_sum
    
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
    txs_len = len(inp)

    global bitvector_dict
    bitvector_dict = {}

    for idx, data in enumerate(inp):
        v = [frozenset([int(i)]) for i in data.rstrip().split()]
        for item in v:
            if item not in bitvector_dict:
                bitvector_dict[item] = np.array([0] * txs_len)
            bitvector_dict[item][idx] = 1 

    global result
    result = []

    start = timer()
    print("Start Timer...")

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
