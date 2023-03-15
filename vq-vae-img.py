import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import numpy as np

import utils
import os


use_cuda = torch.cuda.is_available()

## Model

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.ReLU(), 
                nn.Conv2d(ch, ch, 3), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(ch, ch, 1, padding=1))
    def forward(self, x):
        return self.block(x) + x

def encoder(ch):
    return nn.Sequential( 
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, ch, 4, 2, padding=1),
            ResidualBlock(ch), ResidualBlock(ch))

def decoder(ch):
    return nn.Sequential( 
            ResidualBlock(ch), ResidualBlock(ch),
            nn.ConvTranspose2d(ch, 64, 4, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, padding=1),
            nn.Sigmoid())

## Dataset


dataset = torchvision.datasets.CIFAR10(root="./data/cifar10", download=True, train=True, transform=transforms.ToTensor())


## Hyperparameters

batch_size = 128
lr = 2e-4
# categorical dimention
K = 128
# embedding size
D = 128
beta = 0.25 
total_steps = 250000 * batch_size

def train():
    step = 0
    epochs = total_steps // len(dataset) + 1
    tq = tqdm(range(epochs))
    for epoch in tq:
        for input, _ in DataLoader(dataset, batch_size=batch_size, pin_memory=use_cuda, num_workers=2, shuffle=True, drop_last=True):
            if use_cuda:
                input = input.cuda()

            step += batch_size 
            if step > total_steps:
                return

            ze = enc(input)
            field_size = ze.shape[-1]

            # calculate distance to embeddings
            index = (torch.cdist(
                rearrange(ze.detach(), 'B D W H -> B (W H) D'), 
                repeat(embeddings, 'K D -> B K D', B = batch_size))
                .min(-1)[1])
            sz = index.size()

            # running exponential average
            emb = embeddings.detach() * 0.99 + embeddings.clone() * 0.01

            zq = rearrange(emb[index.view(-1)], '(B W H) D -> B D W H', B=batch_size, W=field_size, H=field_size)

            emb_loss = (zq - ze.detach()).pow(2).sum(1).mean() + 1e-2 * embeddings.pow(2).mean()

            # detach zq so it won't backprop to embeddings with recon loss
            zq = zq.detach()#, requires_grad=True
            zq.requires_grad_(True)

            output = dec(zq)

            commit_loss = beta * (ze - zq.detach()).pow(2).sum(1).mean()
            recon_loss = F.mse_loss(output, input)

            optimizer.zero_grad()

            commit_loss.backward(retain_graph=True)
            emb_loss.backward()
            recon_loss.backward()

            # pass data term gradient from decoder to encoder
            ze.backward(zq.grad)

            optimizer.step()

            emb_count[index.data.view(-1)] = 1
            emb_count.sub_(0.01).clamp_(min=0)
            unique_embeddings = emb_count.gt(0).sum().item()

            tq.set_postfix(
                recon_loss = recon_loss.item(),
                commit_loss = commit_loss.item(),
                unique_embeddings = unique_embeddings,
                step=step)


        input = input.cpu()[0:8,:,:,:]
        w = input.size(-1)
        output = output.data.cpu()[0:8,:,:,:]
        result = (torch.stack([input, output])           # [2, 8, 3, w, w]
                    .transpose(0, 1).contiguous()        # [8, 2, 3, w, w]
                    .view(4, 4, 3, w, w)                 # [4, 4, 3, w, w]
                    .permute(0, 3, 1, 4, 2).contiguous() # [4, w, 4, w, 3]
                    .view(w * 4, w * 4, 3))              # [w * 4, w * 4, 3]

        Image.fromarray( np.uint8((result * 255).numpy())).save("samples/{:03d}.jpg".format(epoch))

## Train

enc = encoder(D) 
dec = decoder(D)
embeddings = torch.randn(K, D).div(D)

# calculate number of embedding vectors used
emb_count = torch.zeros(K)

if use_cuda:
    enc = enc.cuda()
    dec = dec.cuda()
    embeddings = embeddings.cuda()
    emb_count = emb_count.cuda()

embeddings = nn.Parameter(embeddings, requires_grad=True)
optimizer = optim.AdamW([ 
    { 'params' : enc.parameters() },
    { 'params' : dec.parameters() },
    { 'params' : embeddings }
    ], lr=lr)

train()
