import json
import numpy as np

# Reads the coefficients list
with open('coefficients6.txt', 'r') as f:
    coefficients = json.load(f)

# Asks for the refcode
print('What is the molecule\'s reference code?')
refcode = input().upper()

# If the refcode is valid
if len(refcode) == 6 and refcode in coefficients.keys():
    # Use the set of predictors with the highest R^2 value
    coeffs = coefficients[refcode]['predictors'][coefficients[refcode]['r2'].index(max(coefficients[refcode]['r2']))]
    predictors = list(coeffs.keys())
    print(predictors)
    input_values = []
    bond_length = 0
    # Gets the value for each input predictor
    for i in predictors:
        print('What is the value of', i, '?')
        input_values.append(float(input()))
    # Normalize the data
    if len(input_values) > 1:
        input_values = (np.array(input_values) - np.mean(input_values)) / np.std(input_values)
    # Multiply by its coefficient and print the bond length
    for i in predictors:
        bond_length += coeffs[i] * input_values[predictors.index(i)]
    print(bond_length)
else:
    print('Invalid reference code. Please try again')

