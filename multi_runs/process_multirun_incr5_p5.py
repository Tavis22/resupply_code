import h5py
import numpy as np
import pandas as pd

trials = 5
p = 5
runs = 6
incr = 5

log = pd.read_csv('multirun_p' + str(p) + "_incr" + str(incr) + '.csv')
expectation_values = log.fun 
labels = []
labels.append('qwoa_' + str(p) + '_' + str(pd.Series.idxmin(expectation_values[0:p]) + 1))
for i in range(runs):
    labels.append('qwoa_' + str(p+i) + '_' + str(pd.Series.idxmin(expectation_values[(i+1)*trials:(i+2)*trials])-(i+1)*trials + 1))

data = h5py.File("multirun_p" + str(p) + "_incr" + str(incr) + '_initial.h5','r')
quals = np.array(data[labels[0]]["qualities"]).view(np.float64)
unique_quals = sorted(set(quals))
number_unique_quals = len(unique_quals)
data_out = {}
run_labels = ['Initial_p' + str(p)] + ['p = ' + str(p+i) for i in range(runs)]

qual_dict = {i : 0 for i in unique_quals}
state = np.array(data[labels[0]]["final_state"]).view(np.complex128)
probs = np.abs(state)**2
for i in range(len(probs)):
    qual_dict[quals[i]] += probs[i]
distribution = []
for i in qual_dict:
    distribution.append(qual_dict[i])
data_out[run_labels[0]] = distribution

for j in range(runs):
    data = h5py.File("multirun_p" + str(p) + "_incr" + str(incr) + "_run" + str(j + 1) + '.h5','r')
    qual_dict = {i : 0 for i in unique_quals}
    state = np.array(data[labels[j+1]]["final_state"]).view(np.complex128)
    probs = np.abs(state)**2
    for i in range(len(probs)):
        qual_dict[quals[i]] += probs[i]
    distribution = []
    for i in qual_dict:
        distribution.append(qual_dict[i])
    data_out[run_labels[j+1]] = distribution

output = pd.DataFrame(data_out, columns = run_labels, index = unique_quals)

output.to_csv (r'multirun_p' + str(p) + "_incr" + str(incr) + '_qual_distributions.csv', index = True, header=True)
