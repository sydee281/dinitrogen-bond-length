import json
import numpy as np
import matplotlib.pyplot as plt

with open('coefficients6.txt', 'r') as f:
    coefficients = json.load(f)

refcodes = coefficients.keys()
set_names = []
for code in coefficients.keys():
    for set in range(len(coefficients[code]['r2'])):
        set_names.append(code)

descriptors = ['dm2', 'cn2_x', 'cn2_y', 'cn2_z', 'm2n2_angle']

map_array = []
for ref in refcodes:
    print('processing', ref)
    for sets in coefficients[ref]['predictors']:
        temp_list = []
        for p in descriptors:
            if p not in list(sets.keys()):
                temp_list.append(0.0)
            else:
                temp_list.append(sets[p])
        map_array.append(temp_list)
map_array = np.transpose(map_array)
map_numpy = np.array(map_array)
print(len(map_numpy))
fig, ax = plt.subplots()
im = ax.imshow(map_numpy)

# We want to show all ticks...
ax.set_xticks(np.arange(len(set_names)))
ax.set_yticks(np.arange(len(descriptors)))
# ... and label them with the respective list entries
ax.set_xticklabels(set_names)
ax.set_yticklabels(descriptors)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

ax.set_title("Reference Codes vs Descriptors ")
fig.tight_layout()
plt.show()
