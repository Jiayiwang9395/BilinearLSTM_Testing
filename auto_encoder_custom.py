
#%%
### LSTM auto encoder
### return model with feature reconstruction


import torch
from torch import nn
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2, batch_size=16):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)
        self.relu = nn.ReLU()

        """FC layer to Encoder init for classification"""
        self.fc1 = nn.Linear(128,16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)


    def forward(self, x, hidden):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # hidd_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
	    # # c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        # cell_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())

        outputs, hidden = self.lstm(x, hidden)

        """FORWARD PASS THROUGH FC LAYERS HERE"""
        outputs = torch.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        outputs = torch.relu(self.fc2(outputs))
        outputs = self.dropout(outputs)
        outputs = torch.sigmoid(self.fc3(outputs))
        return outputs, (hidden, cell)


class Seq2Seq(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = Encoder(
            input_size = args.input_size,
            hidden_size = args.hidden_size,
            num_layers = args.num_layers,
            batch_size = args.batch_size,
        )

        self.criterion = nn.MSELoss()
        # self.criterion = nn.BCELoss()

    def forward(self, src,trg):

        batch_size, sequence_length, img_size = src.size()

        """I believe hidden should be size (batch, sequence, hidden)"""
        # hidden = (torch.zeros(self.args.num_layers, self.args.batch_size, self.args.hidden_size),torch.zeros(self.args.num_layers, self.args.batch_size, self.args.hidden_size))
        hidden = (torch.zeros(self.args.batch_size, sequence_length, self.args.hidden_size),torch.zeros(self.args.batch_size, sequence_length, self.args.hidden_size))


        output,_hidden = self.encoder(src,hidden)
        trg = trg.float()
        #inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        output = torch.mean(output,axis = 1)
        output = torch.squeeze(output)
        loss = self.criterion(output, trg)
        return loss
