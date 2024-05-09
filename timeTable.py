import math
import numpy as np

data = 'ibm'
tail = '_RandPert_Reset'
# tail = '_RandPert_Asynch1_MAD2'
config = '5c2a'
models = ['FLRSH']
# models = ['FLRHZ']


def get_time(file):
    with open('./logs/' + file + '.log', 'r') as log:
        lines = log.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('Time elapsed:'):
            break

    return line.split(':')[-1].strip()


if data.lower() == 'forest':
    shl = range(2, 13, 10) if config.startswith('10') else range(2, 13, 5)
elif data.lower() == 'ni':
    shl = range(1, 42, 10)
elif data.lower() == 'ibm':
    shl = [1, 81, 171, 251, 341]
else:
    raise Exception('Data source not implemented.')
for sh in shl:
    for m in models:
        if data.lower() == 'forest':
            file1 = 'forest_Sh{0}_{1}{2}{3}'.format(sh, m, config, tail)
        elif data.lower() == 'ni':
            file1 = 'NI+Share{0}_{1}{2}{3}'.format(sh, m, config, tail)
        elif data.lower() == 'ibm':
            file1 = 'IBMU4_Sh{0}_{1}{2}_Decay.01{3}'.format(sh, m, config, tail)
        else:
            raise Exception('Data source not implemented.')

        line1 = get_time(file1)

        print(line1)
