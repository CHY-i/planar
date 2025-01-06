# %%
import numpy as np
import networkx as nx
import stim
from copy import deepcopy

def gen_surface_code(d):
    n = d**2 + (d-1)**2
    m = 2*d*(d-1)
    g = nx.grid_graph([2*d-1, 2*d-1])
    physical_qubits = [(i, j) for i in range(2*d-1) for j in range(2*d-1) if i % 2 == j % 2]
    x_stabs = [(i, j) for i in range(0, 2*d-1, 2) for j in range(2*d-1) if j % 2 == 1]
    z_stabs = [(i, j) for j in range(0, 2*d-1, 2) for i in range(2*d-1) if i % 2 == 1]
    x_logical = [(i, j) for i, j in physical_qubits if j == 0]
    z_logical = [(i, j) for i, j in physical_qubits if i == 0]
    hx = np.zeros([len(x_stabs), n], dtype=np.int8)
    for i, loc in enumerate(x_stabs):
        for qubit in g.neighbors(loc):
            hx[i, physical_qubits.index(qubit)] = 1
    hz = np.zeros([len(z_stabs), n], dtype=np.int8)
    for i, loc in enumerate(z_stabs):
        for qubit in g.neighbors(loc):
            hz[i, physical_qubits.index(qubit)] = 1
    lx = np.array([1 if loc in x_logical else 0 for loc in physical_qubits], dtype=np.int8)
    lz = np.array([1 if loc in z_logical else 0 for loc in physical_qubits], dtype=np.int8)
    return n, m, hx, hz, lx, lz


def gen_rotated_surface_code(d):
    n = d**2
    m = d**2 - 1
    hx = np.zeros([m//2, n], dtype=np.int8)
    for i in range(-1, d):
        for j in range(0, d-1, 2):
            if i == -1:
                hx[(i+1)*(d-1)//2+j//2, [(i+1)*d+j+i%2, (i+1)*d+j+1+i%2]] = 1
            elif i == d-1:
                hx[(i+1)*(d-1)//2+j//2, [i*d+j+i%2, i*d+j+1+i%2]] = 1
            else:
                hx[(i+1)*(d-1)//2+j//2, [i*d+j+i%2, i*d+j+1+i%2, (i+1)*d+j+i%2, (i+1)*d+j+1+i%2]] = 1
    hz = np.zeros([m//2, n], dtype=np.int8)
    for i in range(d-1):
        for j in range(-1, d, 2):
            if j == -1 and i % 2 == 0:
                hz[i*(d+1)//2+(j+1)//2, [i*d+j+1+i%2, (i+1)*d+j+1+i%2]] = 1
            elif j == d-2 and i % 2 == 1:
                hz[i*(d+1)//2+(j+1)//2, [i*d+j+i%2, (i+1)*d+j+i%2]] = 1
            else:
                hz[i*(d+1)//2+(j+1)//2, [i*d+j+i%2, i*d+j+1+i%2, (i+1)*d+j+i%2, (i+1)*d+j+1+i%2]] = 1
    lx = np.array([1 if i % d == 0 else 0 for i in range(n)])
    lz = np.array([1 if i % d == i else 0 for i in range(n)])
    return n, m, hx, hz, lx, lz


def gen_toric_code(d):
    d = d
    n = 2*d**2
    m = 2*d**2 - 2
    hz = np.zeros((d, d, n), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            if j == d-1:
                hz[i, j][[i*2*d+j, ((2*i+2)*d+j)%n, (i*2+1)*d+j, i*2*d+j+1]]=1
                #print(i*2*d+j, ((2*i+2)*d+j)%hzelf.n, (i*2+1)*d+j, i*2*d+j+1)
            else:
                hz[i, j][[i*2*d+j, ((2*i+2)*d+j)%n, (i*2+1)*d+j, (i*2+1)*d+j+1]]=1
                #print(i*2*d+j, ((2*i+2)*d+j)%self.n, (i*2+1)*d+j, (i*2+1)*d+j+1)
    hz = hz.reshape(-1, n)
    hx = np.zeros((d, d, n), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            if i == 0 and j!=0:
                hx[i, j][[i*2*d+j, (i*2+1)*d+j, i*2*d+j-1, n-d+j]]=1
                # print(i*2*d+j, (i*2+1)*d+j, i*2*d+j-1, self.n-d+j)
            elif i!=0 and j == 0:
                hx[i, j][[i*2*d+j, (i*2+1)*d+j, (i*2+1)*d-1, (i*2-1)*d+j]]=1
                # print(i*2*d+j, (i*2+1)*d+j, (i*2+1)*d-1, (i*2-1)*d+j)
            elif i ==0 and j ==0:
                hx[i, j][[i*2*d+j, (i*2+1)*d+j, (i*2+1)*d-1, n-d+j]]=1
                # print(i*2*d+j, (i*2+1)*d+j, (i*2+1)*d-1, self.n-d+j)
            else:
                hx[i, j][[i*2*d+j, (i*2+1)*d+j, i*2*d+j-1, (i*2-1)*d+j]]=1
                # print(i*2*d+j, (i*2+1)*d+j, i*2*d+j-1, (i*2-1)*d+j)
    hx = hx.reshape(-1, n)
    lz0 = np.array([1 if i % d == 0 and (i/d)%2 == 1 else 0 for i in range(n)])
    lz1 = np.array([1 if int(i/d) == 0 else 0 for i in range(n)])
    
    lx0 = np.array([1 if int(i/d) == 1 else 0 for i in range(n)])
    lx1 = np.array([1 if i % d == 0 and (i/d)%2==0 else 0 for i in range(n)])
    lx = np.vstack([lx0, lx1])
    lz = np.vstack([lz0, lz1])
    # print(hx)
    # print(lz0, lz1)
    # # print(lx0, lx1)
    # print(lx@lz.T %2)
    # print((hz@lx.T%2).sum())
    # print((hx@lz.T%2).sum())

    return n, m, hx, hz, lx, lz

def gen_rep_cir(d, r):
    r = r
    d = d- 1 # number of ancilla qubits
    l=d+1
    n = 3*(l-1)*r+l
    mx = (l-1)*(r+1)
    mz = n-mx-1
    assert mz == (2*l-2)*r
    Hz=np.zeros((mz, n))
    Hx_Lx=np.zeros((mx+1, n))
    Lz=np.zeros(n)
    
    E = []
    for i in range(mx):
        if (i+1)%d==0:
            E.append(i)
            if int(i/d) != r:
                E.append((i, i+d-1))
                E.append((i, i+d))

        else:
            E.append((i, i+1))
            if i%d==0: 
                if int(i/d) != r:
                    E.append((i, i+d))
                E.append((i, -1))
            
            elif i%d !=0 and int(i/d) != r:
                E.append((i, i+d-1))
                E.append((i, i+d))
    for i in range(n):
        Hx_Lx[E[i], i] = 1 
    Hx = Hx_Lx[:-1]
    Lx = Hx_Lx[-1]       
    Lz[-l:]=1
    # print(E)

    for i in range(mz):
        a, b = int(i/2), i%2
        if b ==  0:
            if i%(2*l-2) == 0:

                if int(i/(2*l-2)) == r-1:
                    Hz[i, (i+a+1, i+a+2, i+a+3*d+1)] = 1
                    # print(i, i+a+1, i+a+2, i+a+3*d+1)
                else:
                    Hz[i, (i+a+1, i+a+2, i+a+3*d+2)] = 1
                    # print(i, i+a+1, i+a+2, i+a+3*d+2)
            else:
                if i > (r-1)*(2*l-2)+2:
                    # print(i, int((mz-i)/2))
                    Hz[i, (i+a+1, i+a+2, n-1-int((mz-i)/2))] = 1
                    # print(i, i+a+1, i+a+2, n-1-int((mz-i)/2))
                else:
                    Hz[i, (i+a+1, i+a+2, i+a+3*(d-1))] = 1
                    # print(i, i+a+1, i+a+2, i+a+3*(d-1))
        else:
            if i%(2*l-2) == 1:
                Hz[i, (3*a, 3*a+1, 3*a+4)] = 1
                # print(i, 3*a, 3*a+1, 3*a+4)
            elif (i+1)%(2*l-2) == 0:
                if i == mz-1:
                    Hz[i, (3*a, 3*a+2, n-1)] = 1
                    # print(i, 3*a, 3*a+2, n-1)
                else:
                    Hz[i, (3*a, 3*a+2, 3*a+3*d)] = 1
                    # print(i, 3*a, 3*a+2, 3*a+3*d)
            else:
                Hz[i, (3*a, 3*a+2, 3*a+4)] = 1
                # print(i, 3*a, 3*a+2, 3*a+4)


    assert ((Hx@Hz.T)%2).sum() == 0
    assert ((Lx@Hz.T)%2).sum() == 0
    assert ((Lz@Hx.T)%2).sum() == 0
    assert ((Lz@Lx.T)%2).sum() == 1
    return  Hx, Hz, Lx, Lz

class rep_cir():
    def __init__(self, d, r):
        self.d = d # number of data qubits
        self.r = r # number of rounds
        self.n = 3*(d-1)*r+d
        self.m = (d-1)*(r+1)
        self.hx, self.hz, self.lx, self.lz = self.gen_rep_cir() # hx: parity check; hz: dual of hx; lx: logical check; lz: logical error.
        self.pure_error_basis()

    def gen_rep_cir(self):
        r = self.r
        d = self.d - 1 # number of ancilla qubits
        l=d+1
        n = 3*(l-1)*r+l
        mx = (l-1)*(r+1)
        mz = n-mx-1
        assert mz == (2*l-2)*r
        Hz=np.zeros((mz, n))
        Hx_Lx=np.zeros((mx+1, n))
        Lz=np.zeros(n)

        self.E = []

        self.boundary_nodes = []
        for i in range(mx):
            if (i+1)%d==0:
                self.boundary_nodes.append(i)
                self.E.append([i])
                if int(i/d) != r:
                    self.E.append([i, i+d-1])
                    self.E.append([i, i+d])

            else:
                self.E.append([i, i+1])
                if i%d==0: 
                    if int(i/d) != r:
                        self.E.append([i, i+d])
                    self.E.append([-1, i])
                    self.boundary_nodes.append(i)
                
                elif i%d !=0 and int(i/d) != r:
                    self.E.append([i, i+d-1])
                    self.E.append([i, i+d])

        self.single_error = []
        for i in range(n):
            Hx_Lx[self.E[i], i] = 1
            if len(self.E[i]) == 1:
                self.single_error.append(i)
            elif -1 in self.E[i]:
                self.single_error.append(i)
        Hx = Hx_Lx[:-1]
        Lx = Hx_Lx[-1]       
        Lz[-l:]=1
        # print(E)

        for i in range(mz):
            a, b = int(i/2), i%2
            if b ==  0:
                if i%(2*l-2) == 0:

                    if int(i/(2*l-2)) == r-1:
                        Hz[i, (i+a+1, i+a+2, i+a+3*d+1)] = 1
                        # print(i, i+a+1, i+a+2, i+a+3*d+1)
                    else:
                        Hz[i, (i+a+1, i+a+2, i+a+3*d+2)] = 1
                        # print(i, i+a+1, i+a+2, i+a+3*d+2)
                else:
                    if i > (r-1)*(2*l-2)+2:
                        # print(i, int((mz-i)/2))
                        Hz[i, (i+a+1, i+a+2, n-1-int((mz-i)/2))] = 1
                        # print(i, i+a+1, i+a+2, n-1-int((mz-i)/2))
                    else:
                        Hz[i, (i+a+1, i+a+2, i+a+3*(d-1))] = 1
                        # print(i, i+a+1, i+a+2, i+a+3*(d-1))
            else:
                if i%(2*l-2) == 1:
                    Hz[i, (3*a, 3*a+1, 3*a+4)] = 1
                    # print(i, 3*a, 3*a+1, 3*a+4)
                elif (i+1)%(2*l-2) == 0:
                    if i == mz-1:
                        Hz[i, (3*a, 3*a+2, n-1)] = 1
                        # print(i, 3*a, 3*a+2, n-1)
                    else:
                        Hz[i, (3*a, 3*a+2, 3*a+3*d)] = 1
                        # print(i, 3*a, 3*a+2, 3*a+3*d)
                else:
                    Hz[i, (3*a, 3*a+2, 3*a+4)] = 1
                    # print(i, 3*a, 3*a+2, 3*a+4)


        assert ((Hx@Hz.T)%2).sum() == 0
        assert ((Lx@Hz.T)%2).sum() == 0
        assert ((Lz@Hx.T)%2).sum() == 0
        assert ((Lz@Lx.T)%2).sum() == 1
        return  Hx, Hz, Lx, Lz

    def pure_error_basis(self):
        n = self.n
        dm = self.d - 1
        self.pebz = np.zeros((dm*(self.r+1), n))
        self.pebz[self.boundary_nodes, self.single_error] = 1
        
        for i in range(self.m):
            if i not in self.boundary_nodes:
                dis = abs(i - np.array(self.boundary_nodes))
                mdis = np.min(dis)
                eidx = np.argmin(dis)
                didx = self.boundary_nodes[eidx]
                self.pebz[i, self.single_error[eidx]] = 1

                for j in range(mdis):
                    # print('mdis:', mdis)
                    if i > didx:
                        # print(didx+j, didx+j+1)
                        self.pebz[i, np.where(self.hx[[didx+j,didx+j+1]].sum(0)==2)[0]] = 1
                        
                    else:
                        # print(didx-j, didx-j-1)
                        self.pebz[i, np.where(self.hx[[didx-j,didx-j-1]].sum(0)==2)[0]] = 1
        
        assert (self.hx @ self.pebz.T % 2 - np.eye(self.m, self.m)).all() == 0
        # print(self.hx.astype(np.int32))
        # print(self.pebz.astype(np.int32))
        # print(self.hx @ self.pebz.T % 2)

    def reorder(self, dem, rotated_graph=True):
        d=self.d-1
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
        # print(dorder)
        # print(self.E)
        eorder = []
        for i in range(self.n):
            index = self.E.index(E1[i])
            eorder.append(index)
            # for j in range(len(self.E[index])):
            #     if self.E[index][j] >= 0 :
            #         self.E[index][j] = dorder[self.E[index][j]]

        
        if rotated_graph == False:
            self.hx = self.hx[:, eorder]
            self.pebz = self.pebz[:, eorder]
            self.hz = self.hz[:, eorder]
            self.lx = self.hx[:, eorder]
            self.lz = self.hz[:, eorder]

        elif rotated_graph == True:
            self.hx = self.hx[:, eorder]
            self.hx = self.hx[dorder, :]
            self.pebz = self.pebz[:, eorder]
            self.pebz = self.pebz[dorder, :]
            assert (self.hx @ self.pebz.T % 2 - np.eye(self.m, self.m)).all() == 0

            self.hz = self.hz[:, eorder]
            self.lx = self.lx[eorder]
            self.lz = self.lz[eorder]

            assert ((self.hx @ self.hz.T)%2).sum() == 0
            assert ((self.lx @ self.hz.T)%2).sum() == 0
            assert ((self.lz @ self.hx.T)%2).sum() == 0
            assert ((self.lx @ self.lz.T)%2).sum() == 1
            return eorder, dorder



            
if __name__ == '__main__':    
    import stim   
    d, r = 10, 50
    

    circuit = stim.Circuit.generated(code_task="repetition_code:memory",
                                         distance=d,
                                         rounds=r,
                                         after_clifford_depolarization=0.1,
                                         before_measure_flip_probability=0.1,
                                         after_reset_flip_probability=0.1,
                                         )

    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    
   
    rep = rep_cir(d, r)
    
    rep.reorder(dem)
    # print(rep.hx)
        # print('b',E1.index(rep.E[i]))
    # rep.pure_error_basis()


# %%

