import networkx as nx
import numpy as np
from math import atan2, pi, isclose
import scipy
# import mkl
# mkl.set_num_threads(1)

class KacWardSolution:
    def __init__(self, adj_matrix) -> None:
        adj = scipy.sparse.csr_matrix(adj_matrix)
        graph = nx.from_scipy_sparse_array(adj)
        is_planar, _ = nx.check_planarity(graph)
        if not is_planar:
            raise ValueError("The graph is not planar, the KacWard solution does not apply.")
        pos = nx.planar_layout(graph)
        self.n, self.m = graph.number_of_nodes(), graph.number_of_edges()
        di_edge_list = sum([[(i,j), (j,i)] for (i,j) in graph.edges()], start=[])
        self.di_edge_dict = {edge:idx for idx, edge in enumerate(di_edge_list)}
        self.nonbacktrack = scipy.sparse.lil_matrix(
            (2*self.m, 2*self.m), dtype=np.complex128
        ) # the non-backtracking matrix of size 2m x 2m
        for i, j in di_edge_list:
            for k in graph.neighbors(j):
                if k != i:
                    self.nonbacktrack[self.di_edge_dict[(i,j)], self.di_edge_dict[(j,k)]] = \
                        np.exp(1J/2. * compute_angle(pos[i], pos[j], pos[k]))
        pass

    def logZ(self, weights):
        D = scipy.sparse.spdiags(np.tanh(weights), 0, len(weights), len(weights))
        # D = scipy.sparse.lil_matrix(
        #     (2*self.m, 2*self.m), dtype=np.complex128)
        # D[np.arange(2*self.m), np.arange(2*self.m)] = np.tanh(weights)
        # for k, (i,j) in enumerate(self.di_edge_dict.keys()):
        #     D[self.di_edge_dict[(i,j)],self.di_edge_dict[(i,j)]] = np.tanh(weights[k])
        logZ = self.n * np.log(2.0) 
        logZ += logcosh(weights).sum() / 2
        A = scipy.sparse.eye(2*self.m, 2*self.m, dtype=np.complex128) - self.nonbacktrack @ D
        A = scipy.sparse.csc_matrix(A)
        LU = scipy.sparse.linalg.splu(A)
        logdet = np.log(LU.U.diagonal()).sum()#np.sum(np.log(np.diag(LU.U.toarray())))
        comp_part = np.imag(logdet)/pi
        # assert isclose(round(comp_part),comp_part,abs_tol=1e-7)
        logdet = np.real(logdet)
        logZ += 0.5*logdet
        return logZ


def logcosh(x):
    return np.abs(x) + np.log(1+np.exp(-2.0*np.abs(x))) - np.log(2.0)
    
def compute_angle(i,j,k):
    """ return angle difference between (ij) and (jk) (or, in other words, k to j and j to i), giving the 2D coordinates of the points.
    """ 
    k2j = j-k
    j2i = i-j
    return (atan2(k2j[1],k2j[0]) - atan2(j2i[1],j2i[0]) + pi)%(2.0*pi) - pi

def logZ(J):
    J = scipy.sparse.csr_matrix(J)
    epsilon = 1.0e-7
    G = nx.from_scipy_sparse_array(J)
    is_planar, _ = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("The graph is not planar, the KacWard solution does not apply.")
    pos = nx.planar_layout(G)
    m = G.number_of_edges()
    directed_edge_list = sum([[(i,j), (j,i)] for (i,j) in G.edges()], start=[])# + [(j,i) for (i,j) in G.edges()]
    directed_edge_dic = {edge:idx for idx, edge in enumerate(directed_edge_list)}
    B = scipy.sparse.lil_matrix((2*m, 2*m), dtype = np.complex128) # the non-backtracking matrix of size 2m x 2m
    for i, j in directed_edge_list:
        for k in G.neighbors(j):
            if k != i:
                B[directed_edge_dic[(i,j)],directed_edge_dic[(j,k)]] = \
                    np.exp( 1J/2. * compute_angle(pos[i],pos[j],pos[k]))
    D = scipy.sparse.lil_matrix((2*m, 2*m), dtype=np.complex128)
    D[np.arange(2*m), np.arange(2*m)] = np.tanh(np.array([J[i,j] for i, j in directed_edge_list]))
    # for (i,j) in directed_edge_list:
    #     D[directed_edge_dic[(i,j)],directed_edge_dic[(i,j)]] = np.tanh(J[i,j])
    # weights = np.tanh(np.array([J[i,j] for i, j in directed_edge_list]))
    logZ = G.number_of_nodes() * np.log(2.0) 
    for (i,j) in G.edges():
        logZ += logcosh(J[i,j])
    A = scipy.sparse.eye(2*m, 2*m, dtype=np.complex128, format='csc') - B @ D
    LU = scipy.sparse.linalg.splu(A)
    logdet = np.sum(np.log(np.diag(LU.U.toarray())))
    comp_part = np.imag(logdet)/pi
    assert isclose(round(comp_part),comp_part,abs_tol=epsilon), print(comp_part)
    logdet = np.real(logdet)
    logZ += 0.5*logdet
    return logZ

if __name__ == '__main__':
    d = 3
    edges = [(0, i) for i in range(1, d*(d-1), d-1)] + \
        [(0, i) for i in range(d-1, d*(d-1)+1, d-1)] + \
        [(j+i*(d-1), j+(i+1)*(d-1)) for j in range(1, d) for i in range(d-1)] + \
        [(j*(d-1)+i, j*(d-1)+i+1) for j in range(d) for i in range(1, d-1)]
    np.random.seed(0)
    edge_weights = np.random.randn(len(edges)) # np.random.randn(len(edges))
    weight_edges = [(edge[0], edge[1], weight) for edge, weight in zip(edges, edge_weights)]
    # print(weight_edges)
    weight_edges_dict = {
        edge: weight for edge, weight in zip(edges, edge_weights)
    }
    original_graph = nx.Graph()
    original_graph.add_weighted_edges_from(weight_edges)
    adj_original = nx.adjacency_matrix(original_graph).todense()

    import time
    t0 = time.perf_counter()
    logZ_sparse = logZ(adj_original)
    print(time.perf_counter() - t0)

    n = adj_original.shape[0]
    binary_configs = np.array(
        [list(map(int, bin(x)[2:].zfill(n))) for x in range(2**n)]
    ) * 2 - 1
    logZ_exact = np.log(np.exp(-np.einsum('ab, bc, ac->a', binary_configs, -adj_original, binary_configs)/2).sum())
    assert np.allclose(logZ_exact, logZ_sparse), print(logZ_exact, logZ_sparse)

    t0 = time.perf_counter()
    kw_solution = KacWardSolution(adj_original)
    weights = np.array([adj_original[i, j] for i, j in kw_solution.di_edge_dict.keys()])
    logZ_self = kw_solution.logZ(weights)
    print(time.perf_counter() - t0)
    assert np.allclose(logZ_exact, logZ_self), print(logZ_exact, logZ_self)

