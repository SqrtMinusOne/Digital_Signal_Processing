from collections import OrderedDict
import numpy as np
import json
from pprint import pprint

N_b = int(input('Номер бригады: '))

data = OrderedDict()

data['Nb'] = N_b
data['N'] = 30 + N_b % 5
data['T'] = 0.0005 * (1 + N_b % 3)
data['a'] = (-1) ** N_b * (0.8 + 0.005 * N_b)
data['C'] = 1 + N_b % 5
data['w0'] = np.pi / (6 + N_b % 5)
data['m'] = 5 + N_b % 5
data['U'] = N_b
data['n0'] = N_b % 5 + 3
data['n_imp'] = N_b % 5 + 5
data['B_1'] = 1.5 + N_b % 5
data['B_2'] = 5.7 - N_b % 5
data['B_3'] = 2.2 + N_b % 5
data['w_1'] = np.pi / (4 + N_b % 5)
data['w_2'] = np.pi / (8 + N_b % 5)
data['w_3'] = np.pi / (16 + N_b % 5)

data['a_1'] = 1.5 - N_b % 5
data['a_2'] = 0.7 + N_b % 5
data['a_3'] = 1.4 + N_b % 5
data['Mean'] = N_b % 5 + 3
data['Var'] = N_b % 5 + 5

for key, value in data.items():
    print(f'{key:5}: {value}')

save = bool(input('Сохранить? [1/0] '))
if save:
    with open('data.json', 'w') as file:
        json.dump(data, file, indent=4)
