import quop_mpi as qu
import numpy as np
from mpi4py import MPI
from numpy import loadtxt

trials = 3
max_p = 20
file_name = 'sign_flip_1'

q_global = loadtxt('8nodeQualityVector.csv', delimiter=',')

n_soln = len(q_global)

def local_qualities(N, local_i, local_i_offset, seed = None):
    q_slice = q_global[local_i_offset:local_i_offset + local_i]/370
    a = []
    for i in range(len(q_slice)):
        if i % 2 == 0:
            a.append(q_slice[i])
        else:
            a.append(-q_slice[i])
    q_flip = np.array(a)
    return q_flip
    
def mapping(x):
    q_abs = abs(x) 
    return q_abs*370

comm = MPI.COMM_WORLD

rng = np.random.RandomState(1)

def x0(p,seed):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)

qwoa = qu.MPI.qwoa(n_soln, comm, qubits = False)

qwoa.set_initial_state(name="equal")

qwoa.set_quality_mapping(mapping)

qwoa.log_results(file_name, "equal", action = "a")

qwoa.plan()

qwoa.benchmark(
    range(1, max_p + 1),
    trials,
    param_func = x0,
    qual_func = local_qualities,
    param_persist = True,
    filename = file_name,
    label = "qwoa")
qwoa.destroy_plan()