import math
import numpy as np

data = 'forest'
tail = '_Asynch1_MAD2_mu1e-6'


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
    for m in ['FLRHZ']:
    # for m in ['FLRSH', 'FLNSH']:
        if data.lower() == 'forest':
            file1 = 'forest_Sh{0}_{1}10c3a_AdvHztl{2}'.format(sh, m, tail)
        elif data.lower() == 'ni':
            file1 = 'NI+Share' + str(sh) + '_' + m + '10c3a_AdvHztl' + tail
        elif data.lower() == 'ibm':
            file1 = 'IBMU4_Sh' + str(sh) + '_' + m + '10c3a_Decay.01_AdvHztl' + tail
        else:
            raise Exception('Data source not implemented.')

        line1 = get_accs(file1)

        print(line1)
