import struct
import numpy as np

def read_int(buf, pos):
    val = struct.unpack_from('<i', buf, pos[0])[0]
    pos[0] += 4
    return val

def read_doubles(buf, pos, count):
    arr = np.frombuffer(buf, dtype='<f8', count=count, offset=pos[0]).copy()
    pos[0] += count * 8
    return arr

def parse_ccdata(path):
    with open(path, 'rb') as f:
        buf = f.read()

    pos = [0]
    n_eval = read_int(buf, pos)          # leading header integer

    eigen = {}
    for i in range(n_eval - 2):          
        N = i + 3
        K = 2 * N + 8

        vals = read_doubles(buf, pos, K)             # eigenvalues 
        vecs = read_doubles(buf, pos, K * K)         # inverse eigenvectors

        coeffs = []
        for kk in range(3):                          # 3 subdomains 
            coeffs.append(read_doubles(buf, pos, K * 16))  # K*16 each

        iV = vecs.reshape((K, K), order='F')         # iV[i,j] = vecs[i + j*K]
        x = np.zeros((K, 3, 16))
        for kk in range(3):
            block = coeffs[kk].reshape((K, 16), order='F')  # [i + j*K]
            x[:, kk, :] = block

        eigen[N] = {'L': vals, 'iV': iV, 'x': x}

    return eigen

class Eigenstruct:
    def __init__(self, eigen):
        self.L = eigen['L']
        self.iV = eigen['iV']
        self.x = eigen['x']