
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
        self.fc1 = nn.Linear(128,16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        """NEED TO ADD TO FORWARD PASS THROUGH FC LAYERS HERE"""
        outputs = torch.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        outputs = torch.relu(self.fc2(outputs))
        outputs = self.dropout(outputs)
        outputs = torch.sigmoid(self.fc3(outputs))
        return outputs, hidden


# class Decoder(nn.Module):
#
#     def __init__(self, input_size=4096, hidden_size=1024, output_size=4096, num_layers=2):
#         super(Decoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers
#
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
#                             dropout=0.1, bidirectional=False)
#
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(hidden_size, output_size)
#
#
#     def forward(self, x, hidden):
#         # out: tensor of shape (batch_size, seq_length, hidden_size)
#         output, (hidden, cell) = self.lstm(x, hidden)
#         prediction = self.fc(output)
#
#         return prediction, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()

        hidden_size = args.hidden_size
        input_size = args.input_size
        output_size = args.output_size
        num_layers = args.num_layers

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.criterion = nn.MSELoss()
    def forward(self, src,trg):

        batch_size, sequence_length, img_size = src.size()
        encoder_hidden = self.encoder(src)
        output,_hidden = encoder_hidden
        trg = trg.float()
        #inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        output = torch.mean(output,axis = 1)
        output = torch.squeeze(output)
        loss = self.criterion(output, trg)
        return loss
