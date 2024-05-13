from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

configs = {'allgood': {'models': ['FLRSH'], 'data': 'RandPert', 'adv': 15},
           'mab': {'models': ['FLRSH'], 'data': 'RandPert', 'adv': 15},
           'mad': {'models': ['FLRHZ'], 'data': 'RandPert', 'adv': 15}}

for cfg in configs.keys():
    for sh in [12]:
        if sh == 12:
            c = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        else:
            raise Exception('Number of shared features not implemented.')
        shared = [x for x in range(0, 54) if x not in chain(*c)]
        adv = {i: [*range(len(c[i]), len(c[i]) + len(shared))] for i in range(configs[cfg]['adv'])}
        for i in range(len(c)):
            c[i].extend(shared)

        nf = int(sh + (12 - sh) / 20)
        if configs[cfg]['data'] == 'RandPert':
            tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
        elif configs[cfg]['models'] == 'AdvHztl':
            tr_loader, val_loader, te_loader = adv_loader(batch_size=128, c=c, adv=adv, head='Forest10c3a_Sh' + str(sh),
                                                          compress=True)
        else:
            raise Exception('Data source not implemented.')

        for m in configs[cfg]['models']:
            head2 = 'forest_Sh{0}_{1}'.format(sh, m)
            if m == 'FLRSH':
                model = FLRSH(feats=c, nc=20, classes=7)
            elif m == 'FLNSH':
                model = FLNSH(feats=c, nc=20, classes=7)
            elif m == 'FLRHZ':
                model = FLRHZ(feats=c, nf=[sh, nf-sh], nc=20, classes=7)
            else:
                raise Exception('Model not found.')
            opt = torch.optim.Adam(model.parameters())
            loss = nn.CrossEntropyLoss()

            if cfg == 'allgood':
                tail = '20c{0}a_allgood_{1}_Reset'.format(configs[cfg]['adv'], configs[cfg]['data'])
                cmab = fcmab(model, loss, opt, nc=20, n=100, c='allgood', head=head2 + tail,
                             adv_c=[*range(configs[cfg]['adv'])], fix_reset=True)
            elif cfg == 'mab':
                tail = '20c{0}a_{1}_Reset'.format(configs[cfg]['adv'], configs[cfg]['data'])
                cmab = fcmab(model, loss, opt, nc=20, n=100, c='mablin', head=head2 + tail,
                             adv_c=[*range(configs[cfg]['adv'])], fix_reset=True)
            elif cfg == 'mad':
                tail = '20c{0}a_{1}_Asynch1_MAD2'.format(configs[cfg]['adv'], configs[cfg]['data'])
                cmab = fcmab(model, loss, opt, nc=20, n=100, c='mad', head=head2 + tail,
                             adv_c=[*range(configs[cfg]['adv'])], sync=False, ucb_c=2)
            else:
                raise Exception('Config not implemented.')
            cmab.train(tr_loader, val_loader, te_loader)
