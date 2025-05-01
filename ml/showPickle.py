import pickle

# open a file, where you stored the pickled data
file = 'sc.pickle'

# dump information to that file
with open(file,'rb') as file:
    data = pickle.load(file)

# close the file
print(data)





 
