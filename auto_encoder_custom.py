
#%%
### LSTM auto encoder
### return model with feature reconstruction


import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)
        self.relu = nn.ReLU()

        """TO ADD FC layer to Encoder init for classification"""
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)


    def forward(self,x,hidden):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, hidden = self.lstm(x,hidden)
        """NEED TO ADD TO FORWARD PASS THROUGH FC LAYERS HERE"""
        outputs = torch.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        outputs = torch.relu(self.fc2(outputs))
        outputs = self.dropout(outputs)
        outputs = torch.sigmoid(self.fc3(outputs))
        return outputs, hidden

class Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )


        self.criterion = nn.MSELoss()
        #self.criterion = nn.CrossEntropyLoss()

    # def forward(self, src, trg):
    def forward(self, src,trg):

        batch_size, sequence_length, img_size = src.size()
        print("batch_size {}".format(batch_size))
        hidden = (torch.zeros(self.args.num_layers, batch_size, self.args.hidden_size),torch.zeros(self.args.num_layers, batch_size, self.args.hidden_size))
        #hidden = (torch.randn(self.args.num_layers, batch_size, self.args.hidden_size),torch.zeros(self.args.num_layers, batch_size, self.args.hidden_size))
        hidden = tuple(tensor.to(self.args.device) for tensor in hidden)
        encoder_hidden = self.encoder(src,hidden)
        tem_output,_hidden = encoder_hidden
        #print("trg = {}".format(trg))
        print("trg shape {}".format(trg.shape))
        print("trg is {}".format(trg))
        trg = trg.float()
        print("original output shape {}".format(tem_output.shape))
        print("original output is {}".format(tem_output))
        output = tem_output[:,-1,:]
        output = output.squeeze()
        print("output shape {}".format(output.shape))
        print("output is {}".format(output))
        loss = self.criterion(output, trg)
        return loss,output
