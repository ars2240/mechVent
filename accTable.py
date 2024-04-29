import math
import numpy as np

data = 'forest'
# tail = '_allgood_RandPert_Reset'
tail = '_RandPert_Asynch1_MAD2'
# models = ['FLRSH']
models = ['FLRHZ']


def get_accs(file):
    with open('./logs/' + file + '.log', 'r') as log:
        lines = log.readlines()

    return lines[-1].strip()


if data.lower() == 'forest':
    shl = range(2, 13, 10)
elif data.lower() == 'ni':
    shl = range(1, 42, 10)
elif data.lower() == 'ibm':
    shl = [1, 81, 171, 251, 341]
else:
    raise Exception('Data source not implemented.')
for sh in shl:
    for m in models:
        if data.lower() == 'forest':
            file1 = 'forest_Sh{0}_{1}10c3a{2}'.format(sh, m, tail)
        elif data.lower() == 'ni':
            file1 = 'NI+Share{0}_{1}10c3a{2}'.format(sh, m, tail)
        elif data.lower() == 'ibm':
            file1 = 'IBMU4_Sh{0}_{1}10c3a_Decay.01{2}'.format(sh, m, tail)
        else:
            raise Exception('Data source not implemented.')

        line1 = get_accs(file1)

        print(line1)
