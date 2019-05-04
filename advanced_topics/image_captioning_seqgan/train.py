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
from generator import Generator
from discriminator import Discriminator

image_dir = 'data/resized2014'
caption_path = 'data/annotations/captions_train2014.json'
batch_size = 32
vocab_path = 'data/vocab.pkl'
num_workers = 2
crop_size = 224
embedding_size = 256
lstm_size = 512
num_epochs = 100
attention_dim = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    print('vocab_size:', vocab_size)

    dataloader = get_loader(image_dir, caption_path, vocab, 
                            batch_size,
                            crop_size,
                            shuffle=False, num_workers=num_workers)

    generator = Generator(attention_dim, embedding_size, lstm_size, vocab_size)
    generator = generator.to(device)
    generator = generator.train()

    discriminator = Discriminator(vocab_size, embedding_size, lstm_size, attention_dim)
    discriminator = discriminator.to(device)
    discriminator = discriminator.train()

    for i in range(5):
        generator.pre_train(dataloader, 1, vocab)
        discriminator.pre_train(generator, dataloader, 1, vocab)


#    for i in range(5):
#        print("D")
#        discriminator.pre_train(generator, dataloader, 1, vocab, num_batches=100)
#        print("G")
#        generator.ad_train(dataloader, discriminator, vocab, 1, num_batches=20, alpha_c=1.0)

    
train()

