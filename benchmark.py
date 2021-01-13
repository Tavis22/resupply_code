import quop_mpi as qu
import numpy as np
from mpi4py import MPI
from numpy import loadtxt

trials = 3
max_p = 8 

q_global = loadtxt('8nodeQualityVector.csv', delimiter=',')

n_soln = len(q_global)

def local_qualities(N, local_i, local_i_offset, seed = None):
    return q_global[local_i_offset:local_i_offset + local_i]

comm = MPI.COMM_WORLD

rng = np.random.RandomState(1)

def x0(p,seed):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)

qwoa = qu.MPI.qwoa(n_soln, comm, qubits = False)

qwoa.set_initial_state(name="equal")

qwoa.set_optimiser('scipy',{'method':'BFGS','tol':1e-3,'options':{'eps':2e-06}},['fun','nfev','success'])

qwoa.log_results("qwoa_complete_equal", "equal", action = "a")

qwoa.plan()

qwoa.benchmark(
    range(1, max_p + 1),
    trials,
    param_func = x0,
    qual_func = local_qualities,
    param_persist = True,
    filename = "qwoa_complete_equal",
    label = "qwoa")
qwoa.destroy_plan()