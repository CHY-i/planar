# %%
import numpy as np
from scipy.linalg import block_diag
from .utils import error_solver, minmax
from .kacward import KacWardSolution
import networkx as nx
# import time 
def construct_kac_ward_solution(pcm):
    n = pcm.shape[0]
    edges_dict = {}
    for m in range(pcm.shape[1]):
        edge = pcm[:, m].nonzero()[0]
        assert len(edge) <= 2, print('non planar graph')
        if len(edge) == 1:
            if (edge[0], n) not in edges_dict.keys():
                edges_dict[(edge[0], n)] = [m]
            else:
                edges_dict[(edge[0], n)].append(m)
        else:
            assert edge[0] < edge[1]
            if (edge[0], edge[1]) not in edges_dict.keys():
                edges_dict[(edge[0], edge[1])] = [m]
            else:
                edges_dict[(edge[0], edge[1])].append(m)
    g = nx.from_edgelist(edges_dict.keys())
    kwsolution = KacWardSolution(nx.adjacency_matrix(g, nodelist=np.arange(n+1)))
    return kwsolution, edges_dict

class Planar:
    def __init__(self, hx, hz, lx, lz) -> None:
        assert hx.shape[1] == hz.shape[1]
        assert lx.shape[0] == lz.shape[0]
        self.n = hx.shape[1]
        self.hx, self.hz, self.lx, self.lz = hx, hz, lx, lz
        self.stabilizers = block_diag(*(self.hz, self.hx))
        self.logicals = block_diag(*(self.lx, self.lz))
        self.pure_error_basis = self.cal_pure_error_basis()
        self.kwx, self.edges_dict_x = construct_kac_ward_solution(hx)
        self.kwz, self.edges_dict_z = construct_kac_ward_solution(hz)
        pass


    def cal_pure_error_basis(self):
        syndrome_generator = np.eye(self.stabilizers.shape[0], dtype=self.stabilizers.dtype)
        pure_error_basis = np.vstack([
            error_solver(self.stabilizers, syndrome_generator[i]) 
            for i in range(self.stabilizers.shape[0])
        ])
        return pure_error_basis
    
    def decode1(self, syndrome, error_prob, peb):
        pure_error =  block_diag(*peb)[syndrome.astype(bool)].sum(0) % 2
        # print(pure_error.shape)
        assert np.allclose(syndrome, (self.stabilizers @ pure_error) % 2)
        pure_error_x, pure_error_z = pure_error[:self.n], pure_error[self.n:]
        error_rate_x = error_prob[1] + error_prob[3]
        error_rate_z = error_prob[2] + error_prob[3]
        probs_x = [
            log_coset_p(pure_error_x, error_rate_x, self.kwx, self.edges_dict_x), 
            log_coset_p((pure_error_x + self.lx) % 2, error_rate_x, self.kwx, self.edges_dict_x)
        ]
        ind_x = np.argmax(probs_x)
        probs_z = [
            log_coset_p(pure_error_z, error_rate_z, self.kwz, self.edges_dict_z), 
            log_coset_p((pure_error_z + self.lz) % 2, error_rate_z, self.kwz, self.edges_dict_z)
        ]
        ind_z = np.argmax(probs_z)
        recovery_op = pure_error
        if ind_x: recovery_op = (recovery_op + self.logicals[0]) % 2
        if ind_z: recovery_op = (recovery_op + self.logicals[1]) % 2
        return recovery_op

    def decode(self, syndrome, error_prob):
        pure_error = self.pure_error_basis[syndrome.astype(bool)].sum(0) % 2
        # print(pure_error.shape)
        assert np.allclose(syndrome, (self.stabilizers @ pure_error) % 2)
        pure_error_x, pure_error_z = pure_error[:self.n], pure_error[self.n:]
        error_rate_x = error_prob[1] + error_prob[3]
        error_rate_z = error_prob[2] + error_prob[3]
        probs_x = [
            log_coset_p(pure_error_x, error_rate_x, self.kwx, self.edges_dict_x), 
            log_coset_p((pure_error_x + self.lx) % 2, error_rate_x, self.kwx, self.edges_dict_x)
        ]
        # a = log_coset_p1(pure_error_x, error_rate_x, self.kwx, self.edges_dict_x, error_rates=np.random.randn(13))
        ind_x = np.argmax(probs_x)
        probs_z = [
            log_coset_p(pure_error_z, error_rate_z, self.kwz, self.edges_dict_z), 
            log_coset_p((pure_error_z + self.lz) % 2, error_rate_z, self.kwz, self.edges_dict_z)
        ]
        ind_z = np.argmax(probs_z)
        recovery_op = pure_error
        if ind_x: recovery_op = (recovery_op + self.logicals[0]) % 2
        if ind_z: recovery_op = (recovery_op + self.logicals[1]) % 2
        return recovery_op



class Planar_rep_cir:
    def __init__(self, hx, hz, lx, lz) -> None:
        assert hx.shape[1] == hz.shape[1]
        assert lx.shape[0] == lz.shape[0]
        self.n = hx.shape[1]
        self.hx, self.hz, self.lx, self.lz = hx, hz, lx, lz
        self.kwz, self.edges_dict_z = construct_kac_ward_solution(hz)
        pass


    def cal_pure_error_basis(self):
        syndrome_generator = np.eye(self.hx.shape[0], dtype=self.hx.dtype)
        pure_error_basis = np.vstack([
            error_solver(self.hx, syndrome_generator[i]) 
            for i in range(self.hx.shape[0])
        ])
        # print(pure_error_basis)
        return pure_error_basis

    def decode(self, syndrome, error_rates, pebz):
        pure_error_z = pebz[syndrome.astype(bool)].sum(0) % 2
        probs_z = [
            rep_cir_log_coset_p(pure_error_z, self.kwz, self.edges_dict_z, error_rates=error_rates), 
            rep_cir_log_coset_p((pure_error_z + self.lz) % 2, self.kwz, self.edges_dict_z, error_rates=error_rates)
        ]
        ind_z = np.argmax(probs_z)
        
        
        recover = (pure_error_z + self.lz*ind_z)%2
        return recover

    
def exact_coset_probs(pcm, operator, error_rate):
    operator = operator.reshape(-1)
    from scipy.special import logsumexp
    m = pcm.shape[0]
    bitstrings = np.array(
        [list(map(int, bin(x)[2:].zfill(m))) for x in range(2**m)]
    )
    log_ps = np.empty(2**m)
    for s in range(2**m):
        bitstring = bitstrings[s]
        factor = operator.copy()
        for i, g in enumerate(bitstring):
            if g == 1:
                factor += pcm[i]
        log_ps[s] = np.log(np.array([1-error_rate, error_rate])[(factor % 2)]).sum()
    log_coset_p = logsumexp(log_ps)
    return log_coset_p


def log_coset_p(operator, error_rate, solution:KacWardSolution, edges_dict):
    if len(operator.shape) >= 2:
        assert operator.shape[0] == 1, print(operator.shape)
        operator = operator.reshape(-1)
    # n = len(operator)
    weight = np.array([
        0.5*np.log((1-error_rate)/error_rate),
        0.5*np.log(error_rate/(1-error_rate)),
    ])
    # t = time.time()
    weights = weight[operator]
    weights_nonbacktrack = np.array([
        sum(weights[edges_dict[minmax(*edge)]]) for edge in solution.di_edge_dict
    ])
    logZ = solution.logZ(weights_nonbacktrack)
    return logZ
    
def rep_cir_log_coset_p(operator, solution:KacWardSolution, edges_dict, error_rates):
    if len(operator.shape) >= 2:
        assert operator.shape[0] == 1, print(operator.shape)
        operator = operator.reshape(-1)
    
    error_rates = np.array(error_rates).astype(np.float64)
    weight = 0.5*np.log((1-error_rates)/error_rates)
    weights = weight*(1-2*operator)
    weights_nonbacktrack = np.array([
        sum(weights[edges_dict[minmax(*edge)]]) for edge in solution.di_edge_dict
    ])#.astype(np.float64)+1e-10

    # print(repr(weights_nonbacktrack))

    logZ = solution.logZ(weights_nonbacktrack)
    constants = 0.5 * np.log((1-error_rates) * error_rates).sum()
    logZ += constants
    return logZ

def exact_rep_cir(pcm, operator, error_rates):
    operator = operator.reshape(-1)
    from scipy.special import logsumexp
    m = pcm.shape[0]
    bitstrings = np.array(
        [list(map(int, bin(x)[2:].zfill(m))) for x in range(2**m)]
    )
    log_ps = np.empty(2**m)
    for s in range(2**m):
        bitstring = bitstrings[s]
        factor = operator.copy()
        for i, g in enumerate(bitstring):
            if g == 1:
                factor += pcm[i]
        # if (factor%2).sum() == 1:
        #     print(factor%2)
        #     a += 1
        for k  in range(len(error_rates)):
            log_ps[s] += np.log(np.array([1-error_rates[k], error_rates[k]])[(factor[k] % 2)])
    log_coset_p = logsumexp(log_ps)
    return log_coset_p

# def exact_rep_cir_det(pcm, operator, error_rates):
#     operator = operator.reshape(-1)
#     n = pcm.shape[1]
#     bitstrings = np.array(
#         [list(map(int, bin(x)[2:].zfill(m))) for x in range(2**n)]
#     )
#     for s in range(2*n):
#         bitstring = bitstrings[s]
#         factor = operator.copy()
#         for i, g in enumerate(bitstring):
#             if g == 1:
#                 factor += pcm[i]
#         for k  in range(len(error_rates)):
#             log_ps[s] += np.log(np.array([1-error_rates[k], error_rates[k]])[(factor[k] % 2)])
#     log_coset_p = logsumexp(log_ps)
#     return log_coset_p
# %%
