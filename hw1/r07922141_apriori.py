import numpy as np
import itertools
from timeit import default_timer as timer
import sys

def combination(lst, n_comb):
    to_return = []
    for l in list(itertools.combinations(lst, n_comb)):
        to_return.append(frozenset(l))
        
    return list(set(to_return))

# To find all the combination given L(k-1) using union
def unionSet(lst):
    lg = len(lst)
    s_lg = len(lst[0])
    s_l = [l for l in lst]
    u_s = [frozenset(s_l[i].union(s_l[j])) for i in range(lg-1) for j in range(i+1, lg) if len(s_l[i].union(s_l[j])) == s_lg+1]
    
    return list(set(u_s))

# To filter C to L
def C2L(dic_, n_l):
    to_return = {}
    for key, value in dic_.items():
        if value >= n_l:
            to_return[key] = value
    
    return to_return

def C2Scan(_C):
    to_return = {}
    for txd in _C:
        to_return[txd] = TxdCount(txd)
    
    return to_return

# Combination for the new C its subsets are all subset of L
def _C2C(L, lst):
    lf = len(L[0])
    to_return = []
    for value in lst:
        if len(value) > lf + 1:
            continue
            
        tmp = combination(value, lf)
        if set(tmp).issubset(L):
            to_return.append(value)
    
    return to_return

# To build the first database given transaction (input.txt)
def build_DB(Txd):
    to_return = {}
    for idx, data in enumerate(Txd):
        to_return[idx] = set(list(map(int, data.rstrip().split())))
    
    return to_return

# To count the frequency of the given combination
def TxdCount(Txd):
    to_return = 0
    for key, value in DB.items():
        if set(Txd).issubset(value):
            to_return += 1
    
    return to_return

def main(input_path, m_supp, output_path):
    with open(input_path, 'r') as fin:
        inp = fin.readlines()

    # toy_example = ['1 3 4', '2 3 5', '1 2 3 5', '2 5']
    # inp = toy_example

    min_support = len(inp) * m_supp

    start = timer()
    print("Start Timer...")

    global DB 
    DB = build_DB(inp)

    # DB to C1
    fre_dict = {}

    for data in inp:
        lst = list(map(int,data.rstrip().split()))
        for number in lst:
            fzn = frozenset([number])
            if fzn not in fre_dict.keys():
                fre_dict[fzn] = 1
            else:
                fre_dict[fzn] += 1

    C = fre_dict
    all_L = []

    # C1 to L1
    L = C2L(C, min_support)
    all_L.append(L)

    # L1 to C2
    # _C = combination(L.keys(), 2)
    _C = unionSet(list(L.keys()))

    # C2 to scan
    C = C2Scan(_C)

    # C2 to L2
    L = C2L(C, min_support)
    all_L.append(L)

    while True:
        if L == {}:
            break

        # L to C 
        _C = unionSet(list(L.keys()))
        _C = _C2C(list(L.keys()), _C)

        # C to scan
        C = C2Scan(_C)

        # C to L
        L = C2L(C, min_support)

        all_L.append(L)

    end = timer()
    print('Time used: ', end - start)

    answer = []
    for dic in all_L:
        for key, value in dic.items():
            answer.append([sorted(list(key), key=lambda x: x), value])

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