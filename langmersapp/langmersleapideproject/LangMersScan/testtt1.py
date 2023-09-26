import json
import numpy as np
import matplotlib.pyplot as plt

# JSON data string
json_data = '''
{
  "type": "SampleSet",
  ...
  "sample_data": {"data": [[24], [28], [3], [7]], "data_type": "uint32", ...},
  "vectors": {"energy": {"data": [-5.0, -5.0, -5.0, -5.0], ...}, "num_occurrences": {"data": [3, 2, 3, 2], ...}},
  "variable_labels": [1954, 1999, 3700, 3715, 3730],
  ...
}
'''

# Deserialize JSON to Python dictionary
data_dict = json.loads(json_data)

# Unpack samples
samples = np.array(data_dict['sample_data']['data']).flatten()
num_vars = data_dict['num_variables']

# Unpack additional data
energies = data_dict['vectors']['energy']['data']
num_occurrences = data_dict['vectors']['num_occurrences']['data']

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(samples, energies, c=num_occurrences, cmap='viridis')
plt.xlabel('Sample')
plt.ylabel('Energy')
plt.title('Sample Energy vs. Sample')
plt.colorbar(label='Number of Occurrences')
plt.show()
