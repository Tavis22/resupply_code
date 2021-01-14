import h5py
import numpy as np
import pandas as pd

trials = 1
max_p = 13

log = pd.read_csv('qwoa_complete_equal.csv')
expectation_values = log.fun 

label = 'qwoa_' + str(max_p) + '_' + str(pd.Series.idxmin(expectation_values) + 1)

data = h5py.File('qwoa_complete_equal.h5','r')

quals = np.array(data[label]["qualities"]).view(np.float64)

unique_quals = sorted(set(quals))

number_unique_quals = len(unique_quals)

data_out = {}

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

output.to_csv (r'quality_distributions.csv', index = True, header=True)
