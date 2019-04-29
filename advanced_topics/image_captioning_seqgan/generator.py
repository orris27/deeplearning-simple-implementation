import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet101
import torchvision.transforms as T
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(torch.nn.Module):
    def __init__(self, enc_img_size=14):
        super(Encoder, self).__init__()
        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((enc_img_size, enc_img_size))
        self.fine_tune()


    def forward(self, imgs):
        features = self.resnet(imgs)
        features = self.adaptive_pool(features) # (batch_size, 2048, enc_img_size, enc_img_size)
        features = features.permute(0, 2, 3, 1) # (batch_size, enc_img_size, enc_img_size, 2048), enc_img_size=14
        return features

    def fine_tune(self, fine_tune=True): # set some layers to be trainable
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(torch.nn.Module):
    def __init__(self, encoder_dim, lstm_size, attention_dim):
        super(Attention, self).__init__()
        self.encoder_fc = nn.Linear(encoder_dim, attention_dim)
        self.decoder_fc = nn.Linear(lstm_size, attention_dim)
        self.att_fc = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, features, hidden_state):
        '''
            features.shape: (batch_size, num_pixels, encoder_dim)
            hidden_state: (batch_size, lstm_size)

        '''
        att1 = self.encoder_fc(features) # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_fc(hidden_state) # (batch_size, attention_dim)
        
        att = self.att_fc(self.relu(att1 + att2.unsqueeze(1))).squeeze(2) # (batch_size, num_pixels)
        alpha = self.softmax(att) # (batch_size, num_pixels)

        attention_weighted_encoding = (alpha.unsqueeze(2) * features).sum(dim=1) # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Decoder(torch.nn.Module):
    def __init__(self, attention_dim, embedding_size, lstm_size, vocab_size, encoder_dim=2048):
        super(Decoder, self).__init__()

        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.rnn_cell = nn.LSTMCell(self.encoder_dim + embedding_size, lstm_size, bias=True)
        self.h_fc = nn.Linear(self.encoder_dim, lstm_size)
        self.c_fc = nn.Linear(self.encoder_dim, lstm_size)
        self.f_beta = nn.Linear(lstm_size, self.encoder_dim)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()


        self.classifier = nn.Linear(lstm_size, vocab_size)
        self.attention = Attention(encoder_dim, lstm_size, attention_dim)


    def forward(self, features, captions, lengths, device='cuda'):
        '''
            features: (batch_size, enc_image_size, enc_image_size, encoder_dim)
            captions: (batch_size, max_length)
            lengths: (batch_size, )
            
        '''
        # flatten features
        features = features.view(features.size(0), -1, features.size(-1)) # (batch_size, num_pixels, encoder_dim)
        
        # embedding
        embeddings = self.embeddings(captions) # (batch_size, max_length, embedding_size)

        # initialize LSTM states
        mean_features = features.mean(dim=1) # (batch_size, encoder_dim)
        hidden_state = self.h_fc(mean_features) # (batch_size, lstm_size)
        cell_state = self.c_fc(mean_features) # (batch_size, lstm_size)


        #############################################
        # Run LSTM
        #############################################
        decoder_lengths = [length - 1 for length in lengths]

        batch_size = features.size(0)
        num_pixels = features.size(1)
        y_predicted = torch.zeros(batch_size, max(decoder_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decoder_lengths), num_pixels).to(device)
        
        for step in range(max(decoder_lengths)):
            curr_batch_size = sum([l > step for l in decoder_lengths])

            attention_weighted_encoding, alpha = self.attention(features[:curr_batch_size], hidden_state[:curr_batch_size]) # (curr_batch_size, encoder_dim)

            gate = self.sigmoid(self.f_beta(hidden_state[:curr_batch_size])) # (curr_batch_size, encoder_dim)
        
            attention_weighted_encoding = gate * attention_weighted_encoding # (curr_batch_size, encoder_dim)
            hidden_state, cell_state = self.rnn_cell(torch.cat([embeddings[:curr_batch_size, step, :], attention_weighted_encoding], dim=1), (hidden_state[:curr_batch_size], cell_state[:curr_batch_size]))
            y_pred = self.classifier(self.dropout(hidden_state))
            y_predicted[:curr_batch_size, step, :] = y_pred
            alphas[:curr_batch_size, step, :] = alpha

        #return y_predicted, captions, lengths, alphas, sorted_idx
        return y_predicted, captions, decoder_lengths, alphas # Return decoder_lengths to replace the original lengths!!! It is every important beacause 1. we do not consider the output of <eos> 2. avoid the bug of 'RuntimeError: select(): index 25 out of range for tensor of size [25, 32, 9956] at dimension 0'


    def generate(self, features, sos_idx, max_length=30, device='cuda'):
        '''
            features: (enc_image_size, enc_image_size, encoder_dim)
            sos_idx: index of <sos>
        '''
        # flatten features
        features = features.view(1, -1, features.size(-1)) # (batch_size, num_pixels, encoder_dim)
        
        # embedding
        inputs = self.embeddings(torch.Tensor([sos_idx]).long().to(device)) # (batch_size=1, embedding_size)

        # initialize LSTM states
        mean_features = features.mean(dim=1) # (batch_size, encoder_dim)
        hidden_state = self.h_fc(mean_features) # (batch_size, lstm_size)
        cell_state = self.c_fc(mean_features) # (batch_size, lstm_size)


        #############################################
        # Run LSTM
        #############################################
        batch_size = features.size(0)
        num_pixels = features.size(1)
        captions = list()
        for step in range(max_length):
            # get attention
            attention_weighted_encoding, alpha = self.attention(features, hidden_state) # (curr_batch_size, encoder_dim)
            gate = self.sigmoid(self.f_beta(hidden_state)) # (curr_batch_size, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding # (curr_batch_size, encoder_dim)
            hidden_state, cell_state = self.rnn_cell(torch.cat([inputs, attention_weighted_encoding], dim=1), (hidden_state, cell_state))
            y_pred = self.classifier(hidden_state)

            _, y_pred = y_pred.max(1)
            captions.append(y_pred)
            inputs = self.embeddings(y_pred)
        captions = torch.stack(captions, 1)

        return captions

    def generate_beamsearch(self, features, sos_idx, eos_idx, beam_size=3, max_length=30, all_captions=False, device='cuda'):
        '''
            features: (1, enc_image_size, enc_image_size, encoder_dim)
            sos_idx: index of <sos>
        '''
        # flatten features
        features = features.view(1, -1, features.size(-1)) # (1, num_pixels, encoder_dim)
        batch_size = features.size(0)
        num_pixels = features.size(1)
        features = features.expand(beam_size, num_pixels, features.size(-1)) # (beam_size, num_pixels, encoder_dim)
        
        # embedding
        curr_indices = torch.Tensor([sos_idx] * beam_size).long() # (beam_size,)
        inputs = self.embeddings(torch.LongTensor(curr_indices).to(device)) # (beam_size, embedding_size)

        # initialize LSTM states
        mean_features = features.mean(dim=1) # (batch_size, encoder_dim)
        hidden_state = self.h_fc(mean_features) # (batch_size, lstm_size)
        cell_state = self.c_fc(mean_features) # (batch_size, lstm_size)


        #############################################
        # Run LSTM
        #############################################
        captions = list()
        k = beam_size
        captions = torch.LongTensor([[sos_idx]] * beam_size).to(device) # (beam_size, 1)
        scores = torch.zeros(beam_size).float().to(device) # (beam_size, )
        top_k_scores = torch.zeros(k).to(device) # (k). Records the current words' probabilities
        complete_captions = list()
        complete_scores = list()
        for step in range(max_length):
            # get attention
            attention_weighted_encoding, _ = self.attention(features, hidden_state) # (curr_batch_size, encoder_dim)
            gate = self.sigmoid(self.f_beta(hidden_state)) # (curr_batch_size, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding # (curr_batch_size, encoder_dim)

            hidden_state, cell_state = self.rnn_cell(torch.cat([inputs, attention_weighted_encoding], dim=1), (hidden_state, cell_state))
            y_pred = self.classifier(hidden_state) # (k, vocab_size)

            y_pred = torch.log_softmax(y_pred, dim=1) # softmax => probability; log => convert multiplication to addition
            
            scores = top_k_scores.unsqueeze(1).expand_as(y_pred) + y_pred # (k, vocab_size)

            if step == 0: # There is only <sos> in the word list at the first step, so select scores[0]
                top_k_scores, top_k_indices = scores[0].topk(k, 0, True, True)
            else: # There must be 3 words in the list
                top_k_scores, top_k_indices = scores.view(-1).topk(k, 0, True, True) # (k, ) , (k,) # top_k_indices[0] belongs to [0, k* vocab_size)
            
            curr_indices = top_k_indices % self.vocab_size # since we reshape scores to 1D vector containing k numbers of vocab_size # (k,)
            prev_indices = top_k_indices / self.vocab_size # (k, )
            
            captions = torch.cat([captions[prev_indices.tolist()], curr_indices.unsqueeze(1)], dim=1) # (k, step+1)

            #########################################################
            # Remove the sentences that have reached <eos>
            #########################################################
            incomplete_indices = [index for index, word_index in enumerate(curr_indices) if word_index != eos_idx]
            complete_indices = list(set(range(len(curr_indices))) - set(incomplete_indices))
            # update the complete captions
            if len(complete_indices) > 0:
                complete_captions.extend(captions[complete_indices].tolist())
                complete_scores.extend(top_k_scores[complete_indices])
            # select the incomplete sentences
            k -= len(complete_indices)
            if k == 0:
                break

            curr_indices = curr_indices[incomplete_indices]
            hidden_state = hidden_state[incomplete_indices]
            cell_state = cell_state[incomplete_indices]
            captions = captions[incomplete_indices]
            features = features[incomplete_indices]
            top_k_scores = top_k_scores[incomplete_indices]
                    
            inputs = self.embeddings(curr_indices)

        if all_captions:
            return complete_captions 
        else:
            i = complete_scores.index(max(complete_scores))
            caption = complete_captions[i]

            return caption

            


class Generator(torch.nn.Module):
    def __init__(self, attention_dim, embedding_size, lstm_size, vocab_size, encoder_dim=2048, fine_tune_encoder=False, encoder_path='data/encoder_params.pkl', decoder_path='data/decoder_params.pkl'):
        super(Generator, self).__init__()

        # ------------- constants ----------------
        self.log_every = 10
        self.save_every = 100

        self.learning_rate = 1e-3

        # ------------- encoder ----------------
        self.encoder = Encoder()
        self.encoder.fine_tune(fine_tune_encoder)
        self.encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.encoder.parameters()), lr=self.learning_rate) if fine_tune_encoder else None

        # ------------- decoder ----------------
        self.decoder = Decoder(attention_dim, embedding_size, lstm_size, vocab_size)
        self.decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder.parameters()), lr=self.learning_rate)
        
        # ------------- load model ----------------
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        if os.path.exists(self.encoder_path):
            print('Start loading encoder')
            self.encoder.load_state_dict(torch.load(self.encoder_path))
        if os.path.exists(self.decoder_path):
            print('Start loading decoder')
            self.decoder.load_state_dict(torch.load(self.decoder_path))

        self.loss_fn = torch.nn.CrossEntropyLoss()

    
    def pre_train(self, dataloader, num_epochs, alpha_c=1.0):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_steps = len(dataloader)

        for epoch in range(num_epochs):
            for index, (imgs, captions, lengths) in enumerate(dataloader):
                imgs = imgs.to(device)
                captions = captions.to(device)

                features = self.encoder(imgs)
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


    def generate(self, img_path, vocab, translate_flag=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))])
        img = Image.open(img_path)
        imgs = transforms(img).to(device).unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(imgs)

        captions = self.sample(features, vocab) # (batch_size, seq_length)
        caption = captions[0]

        if translate_flag:
            def translate(indices):
                sentences = list()
                for index in indices:
                    word = vocab.idx2word[int(index)]
                    if word == '<eos>':
                        break
                    sentences.append(word)
                return ' '.join(sentences)
            
            return translate(caption) # string: <sos> a man ... tree . # no <eos>, but contains <sos>
        else:
            return caption # list, contains <eos> index

            
    def sample(self, features, vocab):
        captions = list()
        with torch.no_grad(): # Avoid accumulating gradients which might result in out of memory
            for feature in features:
                caption = self.decoder.generate_beamsearch(feature.unsqueeze(0), vocab.word2idx['<sos>'], vocab.word2idx['<eos>'], all_captions=False)
                captions.append(caption)
        
        return captions # (batch_size, seq_length)



