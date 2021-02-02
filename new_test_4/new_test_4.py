import quop_mpi as qu
import numpy as np
from mpi4py import MPI
from numpy import loadtxt

trials = 3
max_p = 35

q_global = loadtxt('8nodeQualityVector.csv', delimiter=',')

n_soln = len(q_global)

def local_qualities(N, local_i, local_i_offset, seed = None):
    return (q_global[local_i_offset:local_i_offset + local_i]/370)

def mapping(x):
   mapped_expectation_value = (((x*370)-350)/80)**55
   return mapped_expectation_value

comm = MPI.COMM_WORLD

rng = np.random.RandomState(1)

def x0(p,seed):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)

qwoa = qu.MPI.qwoa(n_soln, comm, qubits = False)

qwoa.set_initial_state(name="equal")

qwoa.set_objective_mapping(mapping)

qwoa.log_results("new_test_4", "equal", action = "a")

qwoa.plan()

qwoa.benchmark(
    range(1, max_p + 1),
    trials,
    param_func = x0,
    qual_func = local_qualities,
    param_persist = True,
    filename = "new_test_4",
    label = "qwoa")
qwoa.destroy_plan()