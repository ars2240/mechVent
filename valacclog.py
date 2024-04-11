import holoviews as hv
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

accplot = False
ncplot = False
sankey = False
chord = False
binlab = False
first = None
print_iter = True
m = 'FLRSH'
tail = '_Reset'

# for sh in [1, 81, 171, 251, 341]:
# for sh in range(1, 42, 10):
for sh in range(2, 3, 10):
    # file = 'IBMU4_Sh' + str(sh) + '_FLRSH10c3a_Decay.01_RandPert'
    # file = 'NI+Share' + str(sh) + '_FLRSH10c3a_RandPert'
    file = 'forest_Sh{0}_{1}10c3a_AdvHztl{2}'.format(sh, m, tail)

    with open('./logs/' + file + '.log', 'r') as log:
        lines = log.readlines()

    clients = [None] * 2
    b_acc, b_ind, c0_acc, c0_ind, c1_acc, c1_ind, nc = [], [], [], [], [], [], []
    acc_list, c_list, value = [], [], []
    matrix = None
    i = 0
    for line in lines:
        line = line.strip()
        if 'True' in line and line.startswith('{'):
            break
        if 'True' in line:
            clients = line.replace('[', '').replace(']', '').strip().split()
            clients = [client == 'True' for client in clients]
            nc.append(sum(clients))
            if sankey or print_iter:
                if binlab:
                    c_list.append(''.join(['1' if x else '0' for x in clients]))
                else:
                    cl = np.where(clients)[0]
                    cl = [str(x) for x in cl]
                    c_list.append(', '.join(cl))
            if chord and matrix is None:
                matrix = np.array([[0] * len(clients)] * len(clients))
        if line.startswith('current:'):
            acc = float(line.split(',')[0].split()[1])
            acc_list.append(acc)
            if sankey:
                value.extend([acc] * nc[-1])
            if chord:
                res = list(combinations(np.where(clients)[0], 2))
                for ix in res:
                    matrix[ix[0], ix[1]] += acc
            if all(clients):
                b_acc.append(acc)
                b_ind.append(i)
            elif clients[0]:
                c0_acc.append(acc)
                c0_ind.append(i)
            elif clients[1]:
                c1_acc.append(acc)
                c1_ind.append(i)
            i += 1

    if print_iter:
        print('Iter\tConfig\tAcc')
        for i in range(len(acc_list)):
            print('{0}\t{1}\t{2}'.format(i, c_list[i], acc_list[i]))

    if accplot:
        plt.plot(b_ind, b_acc, '-o', color='blue', label='Both')
        plt.plot(c0_ind, c0_acc, '-o', color='orange', label='Client 0')
        plt.plot(c1_ind, c1_acc, '-o', color='green', label='Client 1')
        plt.title('Validation Accuracy')
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('./plots/' + file + '_valacc.png')
        plt.clf()
        plt.close()

        plt.plot(b_ind, b_acc, '-o', color='blue', label='Both')
        plt.plot(c1_ind, c1_acc, '-o', color='green', label='Client 1')
        if file.startswith('NI'):
            plt.ylim(94, 100)
        elif file.startswith('IBM'):
            plt.ylim(92, 96)
        z = np.polyfit(b_ind, b_acc, 1)
        p = np.poly1d(z)
        plt.plot(b_ind, p(b_ind), '--', color='blue')
        z = np.polyfit(c1_ind, c1_acc, 1)
        p = np.poly1d(z)
        plt.plot(c1_ind, p(c1_ind), '--', color='green')
        plt.title('Validation Accuracy')
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('./plots/' + file + '_valacctrend.png')
        plt.clf()
        plt.close()

    if ncplot:
        plt.plot(nc)
        plt.title('Number of Clients')
        plt.xlabel("Iterations")
        plt.ylabel("Number of Clients")
        plt.ylim(0, len(clients))
        plt.savefig('./plots/' + file + '_numclients.png')
        plt.clf()
        plt.close()

    if sankey:
        label, indices = np.unique(c_list, return_inverse=True)
        source, target = indices[:-1], indices[1:]
        # source, target, label = c_list[:-nc[-1]], c_list[nc[0]:], np.unique(c_list)
        # nc = np.array(nc) - 1
        # source, target, value, label = nc[:-1], nc[1:], acc_list[:-1], [*range(1, len(clients)+1)]
        # x, y = np.linspace(0, 1, len(clients)), np.linspace(0, 1, len(nc))
        # x, y = np.meshgrid(x, y)
        if first is not None:
            source, target, value = source[:first], target[:first], value[:first]
        fig = go.Figure(go.Sankey(
            orientation="v",
            textfont=dict(size=8),
            node=dict(pad=50, thickness=5,  # x=x, y=y,
                      line=dict(color="black", width=1),
                      label=label, color="blue"),
            link=dict(arrowlen=10, source=source, target=target, value=value)))
        title = 'All Iterations' if first is None else 'First {0}'.format(first)
        fig.update_layout(font=dict(size=12), title=title)
        # fig.update_layout(font_size=10)
        fname = './plots/' + file + '_sankey'
        fname += '.png' if first is None else '_first{0}.png'.format(first)
        fig.write_image(fname)

    if chord:
        res = list(combinations([*range(len(clients))], 2))
        ch_list = []
        for idx in res:
            ch_list.append([idx[0], idx[1], matrix[idx[0], idx[1]]])
        df = pd.DataFrame(ch_list, columns=['source', 'target', 'value'])
        p = hv.Chord(df)
        p.opts(hv.opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=hv.dim('source').str(),
                             labels='source', node_color=hv.dim('index').str()))
        hv.save(p, './plots/' + file + '_chord.png', fmt='png')
