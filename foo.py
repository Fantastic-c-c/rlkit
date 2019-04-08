from joblib import load

with open('output/point-mass/proto-sac-16z-exp-TEST/extra_data.joblib', 'rb') as f:
    d = load(f)

rint(d)
