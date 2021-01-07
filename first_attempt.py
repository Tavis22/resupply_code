from mpi4py import MPI
import numpy as np
import quop_mpi as qu
import matplotlib.pyplot as plt
import h5py as h5

"inputing the quality vector and the number of solutions"
n_soln = 13

q_global = [112.0, 124.0, 122.0, 132.0, 130.0, 120.0, 109.0, 109.0, 145.0, 153.0, 111.0, 111.0, 132.0]

def local_qualities(N, local_i, local_i_offset, seed = None):
##    q_global = pd.read_csv('q_global.csv')['0'].values
    return q_global[local_i_offset:local_i_offset + local_i]


comm = MPI.COMM_WORLD

n_qubits = 4
p = 3

np.random.seed(1)

def x0(p):
    return np.random.uniform(low = 0, high = 1, size = 2 * p)

qwoa = qu.MPI.qwoa(n_soln, comm, qubits = False)

qwoa.log_results("log", "qwoa", action = "a")


"here i have changed the size of the graph to match the number of solutions rather than the full 4 qubit range"
complete_graph = np.ones(n_soln)
complete_graph[0] = 0 

qwoa.set_graph(complete_graph)

qwoa.set_initial_state(name="equal")


"i dont want random qualities, I want my specific quality vector"
qwoa.set_qualities(local_qualities)

qwoa.plan()
qwoa.execute(x0(p))
qwoa.save("qwoa", "example_config", action = "w")

qwoa.destroy_plan()
qwoa.print_result()

f = h5.File('qwoa.h5', 'r')
final_state = np.array(f["example_config"]["final_state"]).view(np.complex128)
qualities = np.array(f["example_config"]["qualities"]).view(np.float64)

ax1 = plt.gca()
ax2 = ax1.twinx()
state_plot = ax1.plot(np.abs(final_state)**2, '+', label = r'Final quantum state, $|<\vec{t}, \vec{\gamma}|\vec{t}, \vec{\gamma} >|^2$')
qual_plot = ax2.plot(qualities,'*', color = 'red', label = r'Qualities, $\vec{q} = q_i$.')
dots1, labels1 = ax1.get_legend_handles_labels()
dots2, labels2 = ax2.get_legend_handles_labels()
plt.legend(dots1 + dots2, labels1 + labels2)
ax1.set_ylabel("Probability")
ax2.set_ylabel("quality")
ax1.set_xlabel("Quantum State/Possible Solution")
plt.savefig("qwoa_final_state")
plt.close()


