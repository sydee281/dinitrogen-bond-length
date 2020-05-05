import json
import numpy as np
from scipy.stats import linregress
import pandas as pd
from sklearn.linear_model import LinearRegression

with open('data.json', 'r') as f:
    data = json.load(f)

# data normalization
normdata = {}
for refcode, entry in data.items():
    normdata[refcode] = {}
    for key, val in entry.items():
        normdata[refcode][key] = (np.array(val) - np.mean(val)) / np.std(val)
        normdata[refcode][key] = [i for i in normdata[refcode][key]]

# save the normalized data
with open('normdata.json', 'w') as f:
    json.dump(normdata, f)

# single-variant linear regression fit
descriptor = 'dn2'
predictors = ['dm2', 'cn2_x', 'cn2_y', 'cn2_z', 'm2n2_angle']

for refcode, entry in normdata.items():
    print('Now processing', refcode)
    y = entry['dn2']
    # print('single-variate linear regression results:')
    inputs = ['dm2', 'cn2_x', 'cn2_y', 'cn2_z', 'm2n2_angle']
    removed = False
    slopeList = []
    for p in predictors:
        x = entry[p]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        cutoff = 0.95
        if cutoff >= abs(slope):
            print('Removed', p, slope)
            inputs.remove(p)
            removed = True
            if abs(slope) > cutoff - 0.05:
                print("*******************************************")
    print(inputs)

    # check for orthonogality
    df = pd.DataFrame(normdata[refcode], columns=inputs)
    corr = df.corr()
    print('correlation matrix:')
    print(corr)

    if removed and len(inputs) > 0:
        # multi-variant linear regression
        # note that i didn't make any selection of the predictors
        X = [entry[p] for p in inputs]
        X = np.transpose(X)
        reg = LinearRegression()
        fit = reg.fit(X, y)
        print('correlation coefficients: ', fit.coef_)  # print the correlation coefficients
        print('R^2: ', reg.score(X, y),)  # print R^2
    print()
