import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from dataloader import get_loader
from generator import Generator

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

   
generator = Generator(attention_dim, embedding_size, lstm_size, vocab_size)
generator = generator.to(device)
generator = generator.eval()


import torchvision.transforms as T
from PIL import Image
transforms = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
img1 = Image.open('data/surf.jpg')
img1 = transforms(img1).to(device)
feature1 = generator.encoder(img1.unsqueeze(0))
img2 = Image.open('data/giraffe.png')
img2 = transforms(img2).to(device)
feature2 = generator.encoder(img2.unsqueeze(0))

features = torch.stack([feature1, feature2], dim=0)
print(generator.sample(features, vocab))



caption = generator.generate('data/surf.jpg', vocab, True)
print(caption)
caption = generator.generate('data/surf.jpg', vocab, False)
print(caption)
caption = generator.generate('data/giraffe.png', vocab, True)
print(caption)
caption = generator.generate('data/giraffe.png', vocab, False)
print(caption)


