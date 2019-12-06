import pickle

with open('output/dclaw-pose-sim/2019_12_05_00_24_44/sim_policy/1.pkl', 'rb') as f:
    data = pickle.load(f)

# still dont quite understand why we have list of 23 numbers here
# print(len(data))
# print(data[0].keys())
# # print(data[-1]['observations'][-1])
# print(data[0]['observations'][20])
print(data[0].keys())

