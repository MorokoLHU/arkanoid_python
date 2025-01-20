import pickle
import csv
import os
import json

def pickle_to_csv(pickle_file, csv_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        print(data)

def list_pickle_files(folder='.'):
    pickle_files = [f for f in os.listdir(folder) if f.endswith('.pickle')]
    return pickle_files


# List all pickle files in the current folder
pickle_files = list_pickle_files()

if pickle_files:
    print("List of pickle files in the current folder:")
    for file in pickle_files:
        csf_file = file.replace('.pickle', '.csv')
        print('--------------------------------------------')
        pickle_to_csv(file, csf_file)
        break
else:
    print("No pickle files found in the current folder.")
