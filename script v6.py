import json
import numpy as np
from scipy.stats import linregress
import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import combinations


def model_stats(final_set):    # Finds the maximum value in a matrix off the diagonal
    df = pd.DataFrame(normdata[refcode], columns=final_set)
    corr = df.corr()
    X = [entry[p] for p in final_set]
    X = np.transpose(X)
    reg = LinearRegression()
    fit = reg.fit(X, y)
    accuracy = float(reg.score(X, y))
    coeffs = {}
    for d in final_set:
        coeffs[d] = fit.coef_[final_set.index(d)]
    maximum = 0
    for a in corr:
        for b in corr:
            if a == b:
                corr[a][b] = 0
            maximum = max([maximum, abs(corr[a][b])])
    max_corrs = maximum  # maximum off-diagonal element
    results = [accuracy, coeffs, max_corrs]
    return results


cutoff_r2 = 0.80
cutoff_corr = 0.50

# save the normalized data
with open('normdata.json', 'r') as f:
    normdata = json.load(f)

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
        corrSum = 0
        for b in inputs:
            if 0.6 < abs(corr[a][b]) < 1:  # If high correlation, add 1 to counter
                corrSum += 1
        if corrSum == 0:  # Sorts between high and low correlation
            off_limits.append(a)
        else:
            mutable.append(a)

    coeffList = []
    R2list = []
    corrMaxList = []
    if len(mutable) > 0 and len(off_limits) > 0:  # If mutable is not empty, run each combination of its elements
        for a in range(len(mutable) + 1):
            comb = combinations(mutable, a)
            for order in comb:  # For each combination of predictors
                finals = off_limits + list(order)
                stats = model_stats(finals)
                R2list.append(stats[0])
                coeffList.append(stats[1])
                corrMaxList.append(stats[2])
    elif len(mutable) > 0:  # If none of the predictors are highly correlated, checks R^2 of off_limits
        for a in range(1, len(mutable)):
            comb = combinations(mutable, a)
            for order in comb:  # For each combination of predictors
                stats = model_stats(list(order))
                R2list.append(stats[0])
                coeffList.append(stats[1])
                corrMaxList.append(stats[2])
    else:
        stats = model_stats(off_limits + list(order))
        R2list.append(stats[0])
        coeffList.append(stats[1])
        corrMaxList.append(stats[2])

    coefficients[refcode] = {'r2': [], 'corr_max': [], 'predictors': []}
    for i, j in enumerate(zip(R2list, corrMaxList, coeffList)):
        if j[0] >= cutoff_r2 and j[1] <= cutoff_corr:
            coefficients[refcode]['r2'].append(j[0])
            coefficients[refcode]['corr_max'].append(j[1])
            coefficients[refcode]['predictors'].append(j[2].copy())
    if len(coefficients[refcode]['r2']) == 0:  # If it doesn't pass the cutoffs, append the highest R^2 value
        max_index = R2list.index(max(R2list))
        coefficients[refcode]['r2'].append(max(R2list))
        coefficients[refcode]['corr_max'].append(corrMaxList[max_index])
        coefficients[refcode]['predictors'].append(coeffList[max_index])

with open('coefficients6.txt', 'w') as outfile:
    json.dump(coefficients, outfile, indent=4)
