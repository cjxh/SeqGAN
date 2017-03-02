import pickle

with open('save/target_params.pkl', 'rb') as f:
    data = pickle.load(f)
    print data
