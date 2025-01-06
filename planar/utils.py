import numpy as np
from ldpc.mod2 import row_echelon
from scipy.linalg import block_diag
from typing import List
from pymatching import Matching

class pymatching_decoder_warpper:
    def __init__(self, hx, hz) -> None:
        self.hx = hx
        self.hz = hz
        self.n = self.hx.shape[1]
        assert self.n == self.hz.shape[1]
        self.check_x_length = self.hx.shape[0]

    def _set_up_decoder(self, error_prob):
        # decoder for Z errors
        pz = error_prob[2] + error_prob[3]
        self.matching_z = Matching.from_check_matrix(
            self.hx,
            weights=np.log((1-pz)/pz)
        )

        # decoder for X-errors
        px = error_prob[1] + error_prob[3]
        self.matching_x = Matching.from_check_matrix(
            self.hz,
            weights=np.log((1-px)/px)
        )

    def decode(self, syndrome, s):
        syndrome_x, syndrome_z = syndrome[:self.check_x_length], syndrome[self.check_x_length:]
        recovery_z = self.matching_z.decode(syndrome_z)
        recovery_x = self.matching_x.decode(syndrome_x)

        return np.hstack([recovery_x, recovery_z])

def subsamples(ds, d, r, dem):
    n = dem.num_errors
    ns = d-ds+1
    subsample = {}.fromkeys(range(ns))
    
    for i in range(ns):
        subsample[i] = ({'det':[], 'em':[], 'er':[], 'merge_e':[]})
        for k in range(r+1):
            for j in range(ds-1):
                subsample[i]['det'].append(i+j+k*(d-1))

        detectors = subsample[i]['det']
        for j, e in enumerate(dem[:n]):
            if j not in subsample[i]['em']:
                Dec = e.targets_copy()
                if len(Dec) == 1 and str(Dec[0]).startswith('D') :
                    if int(str(Dec[0])[1:]) in detectors:
                        subsample[i]['em'].append(j)
                        subsample[i]['er'].append(e.args_copy()[0])
                else:
                    D0 = int(str(Dec[0])[1:])
                    D1 = int(str(Dec[1])[1:])
                    if str(Dec[0]).startswith('D') and str(Dec[1]).startswith('L') and D0 in detectors:
                        # print(str(Dec[0]), str(Dec[1]))
                        # print(j, 'A')
                        subsample[i]['em'].append(j)
                        subsample[i]['er'].append(e.args_copy()[0])
                    elif str(Dec[0]).startswith('D') and str(Dec[1]).startswith('D'):
                        if D0 in detectors and D1 in detectors:
                            # print(str(Dec[0]), str(Dec[1]))
                            # print(j, 'B')
                            subsample[i]['em'].append(j)
                            subsample[i]['er'].append(e.args_copy()[0])
                        elif  D0 not in detectors and D1 in detectors:
                            if D1-D0==1:
                                # print(str(Dec[0]), str(Dec[1]))
                                # print(j, 'C')
                                subsample[i]['em'].append(j)
                                
                                if D1 == detectors[0]:
                                    subsample[i]['er'].append(e.args_copy()[0])
                                else:
                                    subsample[i]['er'].append(e.args_copy()[0]*(1-dem[j-3*(d-1)+2].args_copy()[0]) + (1-e.args_copy()[0])*dem[j-3*(d-1)+2].args_copy()[0])
                                    subsample[i]['merge_e'].append([(j, j-3*(d-1)+2), (D0, D1), (int(str(dem[j-3*(d-1)+2].targets_copy()[0])[1:]), int(str(dem[j-3*(d-1)+2].targets_copy()[1])[1:]))])         
                        elif  D0 in detectors and D1 not in detectors:
                            if D1-D0==1:
                                # print(str(Dec[0]), str(Dec[1]))
                                # print(j, 'E')
                                subsample[i]['em'].append(j)
                                if D0 == detectors[-1]:
                                    subsample[i]['er'].append(e.args_copy()[0])
                                else:
                                    # print(j, j+2)
                                    subsample[i]['em'].insert(-1, j+1)
                                    subsample[i]['er'].append(dem[j+1].args_copy()[0])
                                    subsample[i]['er'].append(e.args_copy()[0]*(1-dem[j+2].args_copy()[0]) + (1-e.args_copy()[0])*dem[j+2].args_copy()[0])                     
                                    subsample[i]['merge_e'].append([(j, j+2), (D0, D1), (int(str(dem[j+2].targets_copy()[0])[1:]), int(str(dem[j+2].targets_copy()[1])[1:]))])
                    else:
                        None
        # assert len(subsample[i][1]) == 3*(ds-1)*r+ds
        # if i == 0 or i == ns-1:
        #     assert len(subsample[i][5]) == r        
        # else:
        #     # print(len(subsample[i][3]))
        #     assert len(subsample[i][5]) == 2*r
                    # if idx in subsample[i][0] and i not in:

    return subsample

class Data():
    def __init__(self,
                 d,
                 r,
                 file_path_detection,
                 file_path_measurement,
                 file_path_logical_flip) -> None:
        self.distance = d
        self.rounds = r
        self.file_path_detection = file_path_detection
        self.file_path_measurement = file_path_measurement
        self.bits_per_shot = (d - 1) * (r+1)
        self.file_path_logical_flip = file_path_logical_flip


    def parse_b8(self, data: bytes, bits_per_shot: int) -> List[List[bool]]:
        shots = []
        bytes_per_shot = (bits_per_shot + 7) // 8
        for offset in range(0, len(data), bytes_per_shot):
            shot = []
            for k in range(bits_per_shot):
                byte = data[offset + k // 8]
                bit = (byte >> (k % 8)) % 2 == 1
                shot.append(bit)
            shots.append(shot)
        return shots
    
    def parse_b8_fast(self, data: bytes, bits_per_shot: int) -> List[List[bool]]:
        bytes_per_shot = (bits_per_shot + 7) // 8
        
        if not isinstance(data, bytes):
            raise ValueError("Input must be a bytes object.")
        byte_array = np.frombuffer(data, dtype=np.uint8)
        bool_array = np.unpackbits(byte_array).astype(bool).reshape(-1, 8)
        bool_array = bool_array[:, ::-1].reshape(-1, bytes_per_shot * 8)  
        return bool_array[:,0:bits_per_shot]
    
    def parse_01(self, data: str) -> List[List[bool]]:
        shots = []
        for line in data.split('\n'):
            if not line:
                continue
            shot = []
            for c in line:
                assert c in '01'
                shot.append(c == '1')
            shots.append(shot)
        return shots


    def logical_flip(self) -> List[List[bool]]:
        with open(self.file_path_logical_flip, 'rb') as f:
            data = f.read()
            # print(data)
        shots = self.parse_b8_fast(data, 1)
        return shots
    
    def detection(self) -> List[List[bool]]:
        with open(self.file_path_detection, 'rb') as f:
            data = f.read()
            # print(data)
        shots = self.parse_b8_fast(data, self.bits_per_shot)
        return shots
    
    def measurement(self):
        with open(self.file_path_measurement, 'rb') as f:
            data = f.read()
            # print(data)
        shots = self.parse_b8_fast(data, self.bits_per_shot+1)
        return shots

def d2m(obv, det, d, r):
    det = det.reshape((-1, r+1, d-1))
    mea = np.zeros_like(det)
    mea[:, 0, :] = det[:, 0, :]
 
    for i in range(1, r):
        mea[:, i, :] = (det[:, i, :] ^ mea[:, i-1, :])

    mea[:, -1, -1] = (obv.squeeze() ^ mea[:, -2, -1] ^ det[:, -1, -1])%2
    for j in range(1, d-1):
        mea[:, -1, -(j+1)] = (mea[:, -1, -j] ^ mea[:, -2, -(j+1)] ^ det[:, -1, -(j+1)])
    mea = mea.reshape(-1, (d-1)*(r+1))
    mea = np.hstack((mea, obv))
    return mea
        


def generate_error(n, error_prob, rng:np.random.RandomState):
    error = rng.choice([0, 1, 3, 2], n, p=error_prob)
    error_z, error_x = error // 2, (error // 2 + error % 2) % 2
    return np.hstack([error_x, error_z])

def generate_circuit_error(trials, n, error_probs, rng:np.random.RandomState):
    error = []
    for i in range(n):
        error.append(rng.choice([0, 1, 3, 2], trials, p=error_probs[i]).reshape(-1, 1))
    error = np.hstack(error)
    error_z, error_x = error // 2, (error // 2 + error % 2) % 2
    return np.hstack([error_x, error_z])

def cal_syndrome(error, stablilzers):
    return (stablilzers @ error) % 2


def decoding_success(lx, lz, recovery_op, original_error):
    logicals = block_diag(*(lz, lx))
    after_op = recovery_op ^ original_error
    success = ((logicals @ after_op) % 2 == 0).all()
    return success


def minmax(x, y):
    return min(x, y), max(x, y)


def row_echelon_self(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon_self(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    # A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    # A[1:] -= A[0] * A[1:,0:1]
    A[1:] = (A[1:] + A[0] * A[1:,0:1]) % 2

    # we perform REF on matrix from second row, from second column
    B = row_echelon_self(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

def error_solver(H, s):
    '''
    get pure error give syndrome and parity check matrix
    '''
    _, _, transfer_matrix, columns = row_echelon(H, full=True)
    error_columns = transfer_matrix @ s % 2
    pure_error = np.array([0 if i not in columns else error_columns[columns.index(i)] for i in range(H.shape[1])])

    return pure_error

def re_order(d, r, dem, rotated_graph=False):
    d -= 1
    if rotated_graph == False:
        E1 = []
        for i, e in enumerate(dem[:dem.num_errors]):
            Dec = e.targets_copy()
            edge1 = []
            for j in range(len(Dec)):
                D = str(Dec[j])
                if D.startswith('D'):
                    idx = int(D[1:])
                    edge1.append(idx)
                if D.startswith('L'):
                    idx = int(D[1:])
                    edge1.append(-1)
            edge1.sort()
            # order.append(E.index(edge1))
            E1.append(edge1) 
    elif rotated_graph == True:
        dorder = [(d-i%d-1)+int(i/d)*d for i in range(dem.num_detectors)]
        E1 = []
        detectors = []
        for i, e in enumerate(dem[:dem.num_errors]):
            Dec = e.targets_copy()
            edge1 = []
            for j in range(len(Dec)):
                D = str(Dec[j])
                if D.startswith('D'):
                    idx = int(D[1:])
                    row = int(idx/d)
                    cul = d-idx%d-1
                    nidx = d*row+cul
                    
                    edge1.append(nidx)
                    if nidx not in detectors:
                        detectors.append(nidx)
                if D.startswith('L'):
                    idx = int(D[1:])
                    edge1.append(-1)
            edge1.sort()
            # order.append(E.index(edge1))
            E1.append(edge1)


    mx = dem.num_detectors
    order = []
    for i in range(mx):
        if (i+1)%d==0:
            order.append(E1.index([i]))
            if int(i/d) != r:
                order.append(E1.index([i, i+d-1]))
                order.append(E1.index([i, i+d]))

        else:
            order.append(E1.index([i, i+1]))
            if i%d==0: 
                if int(i/d) != r:
                    order.append(E1.index([i, i+d]))
                order.append(E1.index([-1, i]))
            
            elif i%d !=0 and int(i/d) != r:
                order.append(E1.index([i, i+d-1]))
                order.append(E1.index([i, i+d]))
    if rotated_graph == False:
        return order
    elif rotated_graph == True:
        return order, dorder

if __name__ == '__main__':
    from qecsim.models.planar import PlanarCode
    from qecsim.models.generic import DepolarizingErrorModel
    import stim

    d, r = 30, 21

    circuit = stim.Circuit.generated(code_task="repetition_code:memory",
                                        distance=d,
                                        rounds=r,
                                        after_clifford_depolarization=0.1,
                                        before_measure_flip_probability=0.1,
                                        after_reset_flip_probability=0.1,
                                        )


    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    sampler = circuit.compile_sampler() #compile_detector_sampler(seed=int(100*error_prob))

    samples = sampler.sample(shots=10)

    det, obv = circuit.compile_m2d_converter().convert(measurements=samples, separate_observables=True)

    # print(det)
    # print(np.array(det))
    # print(obv)

    mea = d2m(obv, det, d, r)
    # print(mea*1.)
    # print(samples*1.)
    print(abs(mea*1.- samples*1.).sum())


    # d = 5
    # code = PlanarCode(d, d)
    # n, k = code.n_k_d[:2]
    # m = n - k
    # hz, hx = code.stabilizers[:m//2, n:], code.stabilizers[m//2:, :n]
    # lx, lz = code.logical_xs[:, :n], code.logical_zs[:, n:]
    # assert not (hx @ lz.T % 2).any()
    # assert not (hz @ lx.T % 2).any()

    # error_model = DepolarizingErrorModel()
    # error = error_model.generate(code, 0.1)
