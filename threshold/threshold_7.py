import quop_mpi as qu
import numpy as np
from mpi4py import MPI
from numpy import loadtxt

trials = 3
max_p = 20
file_name = 'threshold_7'

q_global = loadtxt('8nodeQualityVector.csv', delimiter=',')

n_soln = len(q_global)

def local_qualities(N, local_i, local_i_offset, seed = None):
    q_slice = q_global[local_i_offset:local_i_offset + local_i]
    q_filt = np.fromiter((0 if i > 233 else 0.5 for i in q_slice),float)
    return q_filt

def mapping(x):
    weighted_quality = -200*x
    return weighted_quality

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