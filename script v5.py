import json
import numpy as np
from scipy.stats import linregress
import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import combinations

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
coefficients = {}
for refcode, entry in normdata.items():
    print('Now processing', refcode)
    y = entry['dn2']

    # single-variate linear regression
    inputs = ['dm2', 'cn2_x', 'cn2_y', 'cn2_z', 'm2n2_angle']
    slopeList = {}
    for p in predictors:
        x = entry[p]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        if abs(slope) < 0.1 and len(inputs) > 1:  # Removes inputs that have a slope < 0.1
            inputs.remove(p)

    # Creates correlation matrix
    df = pd.DataFrame(normdata[refcode], columns=inputs)
    corr = df.corr()

    off_limits = []
    mutable = []
    for a in inputs:  # Checks the correlation between two predictors
        sum = 0
        for b in inputs:
            if 0.9 < abs(corr[a][b]) < 1:  # If high correlation, add 1 to counter
                sum += 1
        if sum == 0:  # Sorts between high and low correlation
            off_limits.append(a)
        else:
            mutable.append(a)

    coeffList = {}
    finals = []
    fit = 0
    if len(mutable) > 0:  # If mutable is not empty, run each combination of its elements
        for a in range(len(mutable)):
            comb = combinations(mutable, a)
            for order in comb:
                finals = off_limits + list(order)
                X = [entry[p] for p in finals]
                X = np.transpose(X)
                reg = LinearRegression()
                fit = reg.fit(X, y)
                coeffList[reg.score(X, y)] = finals  # The key is the R^2 and the value is the predictors
    else:  # If none of the predictors are highly correlated, checks R^2 of off_limits
        finals = off_limits
        X = [entry[p] for p in finals]
        X = np.transpose(X)
        reg = LinearRegression()
        fit = reg.fit(X, y)
        coeffList[reg.score(X, y)] = finals  # The key is the R^2 and the value is the predictors

    temp = {}
    mostAccurate = coeffList[max(coeffList.keys())]  # Finds the max R^2 and returns the coefficients that make it
    for j, p in enumerate(mostAccurate):  # Makes a nested dictionary with coefficients
        coefficients[refcode][p] = fit.coef_[j]
print(coefficients)
with open('coefficients5.txt', 'w') as outfile:
    json.dump(coefficients, outfile, indent=4)
