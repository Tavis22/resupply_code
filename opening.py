import h5py

file = h5py.File('qwoa11.h5','r')

# eigenvalues = file['example_config/eigenvalues']
final_state = file['example_config/final_state']
# initial_phases = file['example_config/initial_phases']
qualities = file['example_config/qualities']


state_coeffs = []
for i in final_state: state_coeffs.append(i)

state_probs = []

for i in state_coeffs:
    prob = 0
    for j in i:
        prob += j**2
    state_probs.append(prob)

quals = []
for i in qualities: quals.append(i)

unique_quals = sorted(set(quals))

qual_dict = {i : 0 for i in unique_quals}

for i in range(len(state_probs)):
    qual_dict[quals[i]] += state_probs[i]

output = []

for i in qual_dict:
    output.append([i, qual_dict[i]])
    
from numpy import savetxt

savetxt("qual_dist11.csv", output, delimiter=",")
