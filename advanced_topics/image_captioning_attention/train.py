import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from dataloader import get_loader
from model import Encoder, Decoder

image_dir = 'data/resized2014'
caption_path = 'data/annotations/captions_train2014.json'
batch_size = 32
vocab_path = 'data/vocab.pkl'
num_workers = 2
crop_size = 224
embedding_size = 256
lstm_size = 512
learning_rate = 1e-3
log_every = 10
num_epochs = 100
encoder_path = 'data/encoder_params.pkl'
decoder_path = 'data/decoder_params.pkl'
save_every = 100
alpha_c = 1.0
attention_dim = 512
fine_tune_encoder = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    print('vocab_size:', vocab_size)

    dataloader = get_loader(image_dir, caption_path, vocab, 
                            batch_size,
                            crop_size,
                            shuffle=True, num_workers=num_workers)

    encoder = Encoder().to(device)
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate) if fine_tune_encoder else None

    decoder = Decoder(attention_dim, embedding_size, lstm_size, vocab_size).to(device)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate)

    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path))
    if os.path.exists(decoder_path):
        decoder.load_state_dict(torch.load(decoder_path))

    loss_fn = torch.nn.CrossEntropyLoss().to(device)


    num_steps = len(dataloader)
    for epoch in range(num_epochs):
        for index, (imgs, captions, lengths) in enumerate(dataloader):
            imgs = imgs.to(device)
            captions = captions.to(device)




            features = encoder(imgs)
            #y_predicted, captions_sorted, lengths_sorted, alphas, sorted_idx = decoder(features, captions, lengths)
            y_predicted, captions, lengths, alphas = decoder(features, captions, lengths)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = captions[:, 1:]
            #targets = torch.zeros(captions.size()).long().to(device)
            #targets[:, :-1] = captions[:, 1:]

            y_predicted, _ = pack_padded_sequence(y_predicted, lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, lengths, batch_first=True)

            loss = loss_fn(y_predicted, targets)
            loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()


            
            #optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()


            loss.backward()

            #optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()
            decoder_optimizer.step()

            if index % log_every  == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, num_epochs, index, num_steps, loss.item(), np.exp(loss.item()))) 
    
            #if index % save_every == 0 and index != 0:
            if index % save_every == 0:
                print('Start saving encoder')
                torch.save(encoder.state_dict(), encoder_path)
                print('Start saving decoder')
                torch.save(decoder.state_dict(), decoder_path)
    
train()

