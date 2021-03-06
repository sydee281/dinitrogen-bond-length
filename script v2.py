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
        normdata[refcode][key] = (np.array(val) - np.mean(val))/np.std(val)
        normdata[refcode][key] = [i for i in normdata[refcode][key]]

# save the normalized data
with open('normdata.json', 'w') as f:
    json.dump(normdata, f)       

# single-variant linear regression fit
descriptor = 'dn2'
predictors = ('dm2', 'cn2_x', 'cn2_y', 'cn2_z', 'm2n2_angle')

for refcode, entry in normdata.items():
    print('Now processing', refcode)
    y = entry[descriptor]
    print('single-variate linear regression results:')
    for p in predictors:
        x = normdata[refcode][p]
        print(p, linregress(x, y))

    # check for orthonogality
    df = pd.DataFrame(normdata[refcode], columns=predictors)
    corr = df.corr()
    print('correlation matrix:')
    print(corr)

    # multi-variant linear regression
    # note that i didn't make any selection of the predictors
    X = [entry[p] for p in predictors]
    X = np.transpose(X)
    reg = LinearRegression()
    fit = reg.fit(X, y)
    print('correlation coefficients: ', fit.coef_)    # print the correlation coefficients
    print('R^2: ', reg.score(X, y))  # print R^2
