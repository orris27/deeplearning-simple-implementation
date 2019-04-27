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
decoder = Decoder(attention_dim, embedding_size, lstm_size, vocab_size).to(device)

print('Start loading models.')
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
encoder.eval()
decoder.eval()

