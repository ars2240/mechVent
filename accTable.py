import math
import numpy as np

data = 'forest'
nc = 20
if nc == 20 and data == 'ibm':
    configs = [{'data': data, 'tail': '_RandPert_Reset', 'config': '20c10a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '20c10a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '20c10a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '20c10a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Reset', 'config': '20c15a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '20c15a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '20c15a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '20c15a', 'models': ['FLRHZ']}]
elif nc == 20 and data == 'forest':
    configs = [{'data': data, 'tail': '_RandPert_Reset', 'config': '20c15a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '20c15a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '20c15a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '20c15a', 'models': ['FLRHZ']}]
elif nc == 20:
    configs = [{'data': data, 'tail': '_RandPert_Reset', 'config': '20c10a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '20c10a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '20c10a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '20c10a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Reset', 'config': '20c15a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '20c15a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '20c15a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '20c15a', 'models': ['FLRHZ']}]
elif nc == 10 and data == 'ibm':
    configs = [{'data': data, 'tail': '_Decay.01_RandPert_Reset', 'config': '10c3a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_Decay.01_allgood_RandPert_Reset', 'config': '10c3a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '10c3a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '10c3a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Reset', 'config': '10c5a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '10c5a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '10c5a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '10c5a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Reset', 'config': '10c7a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '10c7a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '10c7a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '10c7a', 'models': ['FLRHZ']}]
elif nc == 10:
    configs = [{'data': data, 'tail': '_RandPert_Reset', 'config': '10c3a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '10c3a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '10c3a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '10c3a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Reset', 'config': '10c5a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '10c5a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '10c5a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '10c5a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Reset', 'config': '10c7a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '10c7a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '10c7a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '10c7a', 'models': ['FLRHZ']}]
elif nc == 5 and data == 'ibm':
    configs = [{'data': data, 'tail': '_Decay.01_RandPert_Reset', 'config': '5c2a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_Decay.01_allgood_RandPert_Reset', 'config': '5c2a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '5c2a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '5c2a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Reset', 'config': '5c3a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '5c3a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '5c3a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '5c3a', 'models': ['FLRHZ']}]
elif nc == 5:
    configs = [{'data': data, 'tail': '_RandPert_Reset', 'config': '5c2a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '5c2a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '5c2a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '5c2a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Reset', 'config': '5c3a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_allgood_RandPert_Reset', 'config': '5c3a', 'models': ['FLRSH']},
               {'data': data, 'tail': '_RandPert_Asynch1_MAD2', 'config': '5c3a', 'models': ['FLRHZ']},
               {'data': data, 'tail': '_RandPert_Asynch1_Cos0', 'config': '5c3a', 'models': ['FLRHZ']}]


def get_accs(file):
    with open('./logs/' + file + '.log', 'r') as log:
        lines = log.readlines()

    return lines[-1].strip()


for c in configs:
    data, tail, config, models = c['data'], c['tail'], c['config'], c['models']
    if data.lower() == 'forest':
        shl = [12] if config.startswith('20') else [2, 12] if config.startswith('10') else range(2, 13, 5)
    elif data.lower() == 'ni':
        shl = range(1, 42, 20) if config.startswith('20') else range(1, 42, 10)
    elif data.lower() == 'ibm':
        shl = [1, 81, 181, 261, 341] if config.startswith('20') else [1, 81, 171, 251, 341]
    else:
        raise Exception('Data source not implemented.')
    for sh in shl:
        for m in models:
            if data.lower() == 'forest':
                file1 = 'forest_Sh{0}_{1}{2}{3}'.format(sh, m, config, tail)
            elif data.lower() == 'ni':
                file1 = 'NI+Share{0}_{1}{2}{3}'.format(sh, m, config, tail)
            elif data.lower() == 'ibm':
                # file1 = 'IBMU4_Sh{0}_{1}{2}_Decay.01{3}'.format(sh, m, config, tail)
                file1 = 'IBMU4_Sh{0}_{1}{2}{3}'.format(sh, m, config, tail)
            else:
                raise Exception('Data source not implemented.')

            line1 = get_accs(file1)

            print(line1)
    print()
