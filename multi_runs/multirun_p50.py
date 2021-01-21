import quop_mpi as qu
import numpy as np
from mpi4py import MPI
from numpy import loadtxt
import pandas as pd
import h5py


trials = 1
p = 50
runs = 4

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

# qwoa.set_optimiser('scipy',{'method':'BFGS','tol':6e-6,'options':{'eps':1e-10, 'gtol':6e-8}},['fun','nfev','success'])

qwoa.log_results("multirun_p"+str(p), "equal", action = "a")

qwoa.plan()

for i in range(runs):

    qwoa.benchmark(
        range(p, p + 1),
        trials,
        param_func = x0,
        qual_func = local_qualities,
        param_persist = False,
        filename = "multirun_p" + str(p) + "_run" + str(i + 1),
        label = "qwoa")
    
    log = pd.read_csv('multirun_p' + str(p) + '.csv')
    expectation_values = log.fun 
    label = 'qwoa_' + str(p) + '_' + str(pd.Series.idxmin(expectation_values[i*trials:(i + 1) * trials])-i*trials + 1)
    data = h5py.File("multirun_p" + str(p) + "_run" + str(i + 1) + '.h5','r')
    nextstate = np.array(data[label]["final_state"]).view(np.complex128)
    qwoa.set_initial_state(state=nextstate)
    
qwoa.destroy_plan()