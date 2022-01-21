import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, sequence_length, hidden_size, num_layers, use_gpu=False):
        super(LSTM, self).__init__()
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        # 5 predictors
        self.input_size = 5
        # binary classification
        self.num_classes = 2
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*sequence_length, self.num_classes)
        # references: https://discuss.pytorch.org/t/sequential-and-static-features-combined-in-lstm-architecture/91115
        self.fc2 = nn.Linear(3, self.num_classes)
        self.fc3 = nn.Linear(4, self.num_classes)
        self.batch_size = None
        self.hidden_cell = None

    def init_hidden(self, x):
        self.batch_size = x.size()[0]
        self.hidden_cell = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double,
                                        device=self.device),
                            torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double,
                                        device=self.device))

    def forward(self, x_time, x_stat):
        x_time, x_stat = x_time.double(), x_stat.double()
        self.init_hidden(x_time)
        print(self.hidden_cell[0].type())
        # input data x
        # can use multiple inputs to forward method: https://discuss.pytorch.org/t/multiple-input-model-architecture/19754
        # for the view call: batch size, sequence length, cols
        lstm_out, self.hidden_cell = self.lstm(x_time.view(self.batch_size, x_time.size()[1], -1), self.hidden_cell)
        # stat_out = stat_input.reshape(self.batch_size,-1)

        stat_preds = self.fc2(x_stat.reshape(self.batch_size, -1))
        time_preds = self.fc1(lstm_out.reshape(self.batch_size, -1))
        preds=torch.cat((stat_preds, time_preds), 1)
        preds = self.fc3(preds)
        return preds.view(self.batch_size, -1)