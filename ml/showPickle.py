import pickle
import re

# open a file, where you stored the pickled data
file = 'scen_info_2025_03_19_22_03_27.pickle'

# load the pickled data
with open(file, 'rb') as f:
    data = pickle.load(f)

# convert data to string if not already
data_str = str(data)

# extract and print each block enclosed in {}
matches = re.findall(r'\{.*?\}', data_str)

for match in matches:
    print(match)
