import torch
import numpy as np
from kernel import Kernel


class shap(object):
    def __init__(self, model, fname, alpha=1e-5, max_iter=50, use_gpu=False):

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)  # model (from torch)
        self.model_load(fname)  # load model state
        self.dataloader = None  # dataloader for training set (from torch)
        self.alpha = alpha  # optimizer function, optional (from torch)
        self.max_iter = max_iter  # maximum number of iterations
        self.v = None  # eigenvector

    def random_v(self, ndim):
        return torch.rand(ndim).to(self.device)

    def load_state(self, fname, dic='state_dict'):

        try:
            state = torch.load(fname, map_location=self.device)
        except RuntimeError:
            state = torch.jit.load(fname)
        if dic in state.keys():
            state2 = state[dic]

            from collections import OrderedDict
            state = OrderedDict()

            for k, v in state2.items():
                k = k.replace('encoder.', 'features.')
                k = k.replace('module.', '')
                p = re.compile("(norm|conv)\.([0-9+])")
                k = p.sub(r'\1\2', k)
                state[k] = v
        return state

    def model_load(self, fname):
        # load model from file
        state = self.load_state(fname)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def f(self, x):
        return torch.norm(x - self.model(self.v), dim=1) ** 2

    def train(self, loader):
        self.dataloader = loader
        outputs = []
        mean = None
        print('epoch\t loss')
        for i in range(self.max_iter):
            for _, data in enumerate(loader):
                print(data.shape)
                input, _ = data
                input = input.to(self.device)

                # initialize v
                if self.v is None:
                    self.v = self.random_v(np.prod(input.shape[1:]))
                    self.v.requires_grad_()

                # compute forward & backward pass
                output = self.model(input)
                loss = self.f(output).sum()
                loss.backward()
                if mean is None:
                    outputs.append(output)

                # get step size
                alpha = self.alpha(i) if callable(self.alpha) else self.alpha

                # take step
                self.v.data -= alpha * self.v.grad.data  # adjust gradient

            # compute mean loss
            if mean is None:
                mean = torch.mean(torch.cat(outputs), dim=0)

            # evaluate neutral instance
            # print(self.v)
            print('%d\t %f' % (i, self.f(mean).item()))

    def shap(self, input):
        # compute shap scores on a batch
        output = self.model(input)

        lv = len(self.v)

        shap = torch.zeros((input.shape[0], lv))
        with torch.no_grad():
            for i in range(lv):
                input2 = input.view(-1, lv)
                input2[:, i] = self.v[i]
                output2 = self.model(input2.view(input.shape))
                shap[:, i] = (output2 - output).mean(dim=1)

        return shap.view(input.shape)

    def test(self, loader=None):
        loader = self.dataloader if loader is None else loader
        shap = []
        for _, data in enumerate(loader):
            input, _ = data
            shap.append(self.shap(input))

        shap = torch.cat(shap)

        """
        import matplotlib.pyplot as plt
        plt.hist(shap.flatten(), bins=20)
        plt.savefig('./shap_hist.png')
        """

        print(shap.mean(dim=0))

    def run(self, train_loader, test_loader=None):
        self.train(train_loader)
        self.test(test_loader)

    def test_explainer(self, loader=None):
        loader = self.dataloader if loader is None else loader

        with torch.no_grad():
            sh = []
            e = Kernel(self.model, self.v.view(1, -1).detach().numpy())
            for _, data in enumerate(loader):
                input, _ = data
                input = input.to(self.device).view(input.shape[0], -1)
                sh.append(e.shap_values(input.detach().numpy()))

        sh = np.concatenate(sh, axis=1)
        if sh.ndim == 3:
            sh = np.moveaxis(sh, [0, 1, 2], [2, 0, 1])
            sh = np.reshape(sh, (sh.shape[0], -1))

        #"""
        import matplotlib.pyplot as plt
        plt.hist(sh.flatten(), bins=20)
        plt.savefig('./shap_hist.png')
        #"""

        return sh

    def explainer(self, train_loader, loader):
        self.train(train_loader)
        sh = self.test_explainer(loader)
        np.savetxt("shap_values.csv", sh, delimiter=",")


