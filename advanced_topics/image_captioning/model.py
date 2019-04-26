import torch
import torch.nn as nn
from torchvision.models import resnet152
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(torch.nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.resnet = resnet152(pretrained=True)
        del self.resnet.fc
        self.resnet.fc = lambda x: x
        self.fc = nn.Linear(2048, embedding_size)
        
        self.bn = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, imgs):
        with torch.no_grad():
            features = self.resnet(imgs)
        features = self.fc(features)
        features = self.bn(features)
        return features



class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, lstm_size, 1, batch_first=True)

        self.classifier = nn.Linear(lstm_size, vocab_size)


    def forward(self, features, captions, lengths):
        embeddings = self.embeddings(captions)
        vec = torch.cat((features.unsqueeze(1), embeddings), 1)
        vec = pack_padded_sequence(vec, lengths, batch_first=True) # 1st param is vec!!! not embeddings!!!
        hiddens, states = self.rnn(vec)
        outputs = self.classifier(hiddens[0]) # [0] to unpack
        return outputs

    def generate(self, features):
        states = None
        inputs = features.unsqueeze(1)
        captions = list()
        for i in range(30):
            hiddens, states = self.rnn(inputs, states)
            outputs = self.classifier(hiddens.squeeze(1))
            _, pred = outputs.max(1)
            captions.append(pred)
            inputs = self.embeddings(pred).unsqueeze(1)
        captions = torch.stack(captions, 1)
        return captions
