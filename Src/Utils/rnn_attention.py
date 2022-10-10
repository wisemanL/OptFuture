import torch
import torch.nn as nn


device = "cpu"

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, int(self.output_size *0.5))

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1) #input: torch.size([1,1]) embedded : (1,1,128)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #embedded[0] : (1,128) #hidden[0] : (1,128) #torch.cat() : (1,256)
        #softmax's dim =1 means applying soft max over "256"
        #attn_weights : (1,10=max_length)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        #attn_weights.unsqueeze(0) : (1,10) -> (1,1,10)
        #encoder_outputs.unsqueeze(0) : (10,128) -> (1,10,128)
        #attn_applied : (1,1,128)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        #output : (1,256=2*hs)
        output = self.attn_combine(output).unsqueeze(0)
        # output : (1,1,hs)

        output = F.relu(output)  # output : (1,1,hs)
        output, hidden = self.gru(output, hidden)
        # output : (1,1,hs) , hidden: (1,1,hs)
        output = F.sigmoid(self.out(output[0]))
        # ouput : (1,2803)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



