 
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

<<<<<<< HEAD
        """TO ADD FC layer to Encoder init for classification"""
        self.fc1 = nn.Linear(hidden_size, 16)
=======
        """FC layer to Encoder init for classification"""
        self.fc1 = nn.Linear(128,16)
>>>>>>> 169750c0c73a2f1ad65a00f61a8ce8327af939c7
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)


<<<<<<< HEAD
    def forward(self, x,hidden):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, hidden = self.lstm(x,hidden)
        """NEED TO ADD TO FORWARD PASS THROUGH FC LAYERS HERE"""
=======
    def forward(self, x, hidden):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # hidd_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
	    # # c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        # cell_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())

        outputs, hidden = self.lstm(x, hidden)

        """FORWARD PASS THROUGH FC LAYERS HERE"""
>>>>>>> 169750c0c73a2f1ad65a00f61a8ce8327af939c7
        outputs = torch.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        outputs = torch.relu(self.fc2(outputs))
        outputs = self.dropout(outputs)
        outputs = torch.sigmoid(self.fc3(outputs))
<<<<<<< HEAD
        return outputs, hidden
=======
        return outputs, (hidden, cell)

>>>>>>> 169750c0c73a2f1ad65a00f61a8ce8327af939c7

class Seq2Seq(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
<<<<<<< HEAD
        self.encoder = Encoder(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )


        self.criterion = nn.MSELoss()
        # self.bce_crit = nn.BCELoss()

    # def forward(self, src, trg):
    def forward(self, src,trg):

        batch_size, sequence_length, img_size = src.size()
        print("batch_size {}".format(batch_size))
        hidden = (torch.zeros(self.args.num_layers, batch_size, self.args.hidden_size),torch.zeros(self.args.num_layers, batch_size, self.args.hidden_size))
        #hidden = (torch.zeros(batch_size,sequence_length,self.args.hidden_size),torch.zeros(batch_size,sequence_length,self.args.hidden_size))
        hidden = tuple(tensor.to(self.args.device) for tensor in hidden)
        encoder_hidden = self.encoder(src,hidden)
        tem_output,_hidden = encoder_hidden
        #print("trg = {}".format(trg))
        print("trg shape {}".format(trg.shape))
        print("trg is {}".format(trg))
=======

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
>>>>>>> 169750c0c73a2f1ad65a00f61a8ce8327af939c7
        trg = trg.float()
        output = tem_output[:,-1,:]
        output = output.squeeze()
        print("output shape {}".format(output.shape))
        print("output is {}".format(output))
        loss = self.criterion(output, trg)
        return loss,output
