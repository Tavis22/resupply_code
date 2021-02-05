import h5py
import numpy as np
import pandas as pd

trials = 3
max_p = 20
file_name = 'threshold_2'

log = pd.read_csv(file_name + '.csv')
expectation_values = log.fun 
labels = []
for i in range(max_p):
    labels.append('qwoa_' + str(i + 1) + '_' + str(pd.Series.idxmin(expectation_values[i*trials:(i + 1) * trials])-i*trials + 1))

data = h5py.File(file_name + '.h5','r')

quals = np.array(data[labels[0]]["qualities"]).view(np.float64)

unique_quals = sorted(set(quals))

number_unique_quals = len(unique_quals)

data_out = {}

for label in labels:
    qual_dict = {i : 0 for i in unique_quals}
    state = np.array(data[label]["final_state"]).view(np.complex128)
    probs = np.abs(state)**2
    for i in range(len(probs)):
        qual_dict[quals[i]] += probs[i]
    distribution = []
    for i in qual_dict:
        distribution.append(qual_dict[i])
    data_out[label] = distribution

output = pd.DataFrame(data_out, columns = labels, index = unique_quals)

output.to_csv(r'quality_distribution_' + file_name + '.csv', index = True, header=True)
