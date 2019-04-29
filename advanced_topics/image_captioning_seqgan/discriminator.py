import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet101
import torchvision.transforms as T
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Discriminator(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout=0.2):
        super(Discriminator, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        self.hidden2out = nn.Linear(hidden_size, 1)


    def forward(self, inputs, hidden):
        '''
            inputs: (batch_size, max_length)
        '''
        inputs = self.embeddings(inputs) # (batch_size, max_length, embedding_size)
        _, hidden = self.rnn(inputs) # hidden: (batch_size, 4, hidden_size)
        hidden = hidden.contiguous() # (batch_size, 4, hidden_size)
        outputs = self.gru2hidden(hidden.view(-1, 4 * self.hidden_size)) # (batch_size, hidden_size)
        
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        outputs = self.hidden2out(outputs) # (batch_size, 1)
        outputs = torch.sigmoid(outputs) # (batch_size, 1)
        

        return outputs.squeeze(1) # (batch_size,)

    def pre_train(self, generator, dataloader, num_epochs, vocab, alpha_c=1.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_steps = len(dataloader)

        for epoch in range(num_epochs):
            for index, (imgs, captions, lengths) in enumerate(dataloader):
                imgs = imgs.to(device)
                captions = captions.to(device) # (batch_size, batch_max_length)

                features = generator.encoder(imgs)
                tmp_captions_pred = generator.samples(features, vocab) # list, (batch_size, var_length)

                
                captions_pred = torch.zeros(captions.size()).to(device)
                lengths_pred = list()
                for i in range(captions.size(0)):
                    captions_pred[i] = tmp_captions_pred[i]
                    lengths_pred = len(tmp_captions_pred[i]) if len(tmp_captions_pred[i]) < captions.size(1) else captions.size(1)

                captions, _ = pack_padded_sequence(captions, lengths, batch_first=True)
                


                captions_pred, _ = pack_padded_sequence(captions_pred, lengths_pred, batch_first=True)
                
                
                
                





                y_predicted, captions, lengths, alphas = self.decoder(features, captions, lengths)

                targets = captions[:, 1:]

                y_predicted, _ = pack_padded_sequence(y_predicted, lengths, batch_first=True)
                targets, _ = pack_padded_sequence(targets, lengths, batch_first=True)

                loss = self.loss_fn(y_predicted, targets)
                loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
                
                #optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()


                loss.backward()

                #optimizer.step()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                if index % self.log_every  == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, num_epochs, index, num_steps, loss.item(), np.exp(loss.item()))) 
        
                if index % self.save_every == 0 and index != 0:
                    print('Start saving encoder')
                    torch.save(self.encoder.state_dict(), self.encoder_path)
                    print('Start saving decoder')
                    torch.save(self.decoder.state_dict(), self.decoder_path)

        
        
