import quop_mpi as qu
import numpy as np
from mpi4py import MPI
from numpy import loadtxt

trials = 2
p = 3

q_global = loadtxt('8nodeQualityVector.csv', delimiter=',')

n_soln = len(q_global)

def local_qualities(N, local_i, local_i_offset, seed = None):
    return q_global[local_i_offset:local_i_offset + local_i]/370

comm = MPI.COMM_WORLD

rng = np.random.RandomState(1)

def x0(p,seed):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)

qwoa = qu.MPI.qwoa(n_soln, comm, qubits = False)

qwoa.set_initial_state(name="equal")

qwoa.log_results("log_basin_tests", "equal", action = "a")

qwoa.plan()

qwoa.set_optimiser('scipy_basinhopping',
                   {'niter':100,'stepsize':0.1,'minimizer_kwargs':{'method':'Nelder-Mead','tol':1e-3,'options':{'maxfev':5000,'maxiter':5000}}},
                   ['fun','nfev','nit'])

qwoa.benchmark(
    range(p, p + 1),
    trials,
    param_func = x0,
    qual_func = local_qualities,
    param_persist = True,
    filename = "test_1",
    label = "qwoa")

qwoa.set_optimiser('scipy_basinhopping',
                   {'niter':150,'stepsize':0.1,'minimizer_kwargs':{'method':'Nelder-Mead','tol':1e-3,'options':{'maxfev':5000,'maxiter':5000}}},
                   ['fun','nfev','nit'])

qwoa.benchmark(
    range(p, p + 1),
    trials,
    param_func = x0,
    qual_func = local_qualities,
    param_persist = True,
    filename = "test_2",
    label = "qwoa")

qwoa.set_optimiser('scipy_basinhopping',
                   {'niter':100,'stepsize':0.01,'minimizer_kwargs':{'method':'Nelder-Mead','tol':1e-3,'options':{'maxfev':5000,'maxiter':5000}}},
                   ['fun','nfev','nit'])

qwoa.benchmark(
    range(p, p + 1),
    trials,
    param_func = x0,
    qual_func = local_qualities,
    param_persist = True,
    filename = "test_3",
    label = "qwoa")

qwoa.set_optimiser('scipy_basinhopping',
                   {'niter':100,'stepsize':0.1,'minimizer_kwargs':{'method':'Nelder-Mead','tol':1e-5,'options':{'maxfev':5000,'maxiter':5000}}},
                   ['fun','nfev','nit'])

qwoa.benchmark(
    range(p, p + 1),
    trials,
    param_func = x0,
    qual_func = local_qualities,
    param_persist = True,
    filename = "test_4",
    label = "qwoa")


qwoa.destroy_plan()