import quop_mpi as qu
import numpy as np
from mpi4py import MPI
from numpy import loadtxt

trials = 40
max_p = 20

q_global = loadtxt('8nodeQualityVector.csv', delimiter=',')

n_soln = len(q_global)

def local_qualities(N, local_i, local_i_offset, seed = None):
    return (q_global[local_i_offset:local_i_offset + local_i]/370)

def mapping(x):
   weighted_quality = -(3**(254-370*x))
   return weighted_quality

comm = MPI.COMM_WORLD

rng = np.random.RandomState(1)

def x0(p,seed):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)

qwoa = qu.MPI.qwoa(n_soln, comm, qubits = False)

qwoa.set_initial_state(name="equal")

qwoa.set_quality_mapping(mapping)

qwoa.log_results("expectation_of_weighted_test_2", "equal", action = "a")

qwoa.plan()

qwoa.benchmark(
    range(max_p, max_p + 1),
    trials,
    param_func = x0,
    qual_func = local_qualities,
    param_persist = True,
    filename = "expectation_of_weighted_test_2",
    label = "qwoa")
qwoa.destroy_plan()