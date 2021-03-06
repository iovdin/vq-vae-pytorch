import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import tensor_comprehensions as tc
import numpy

import lera
import argparse
import utils
import os

parser = argparse.ArgumentParser(description='PyTorch VQ-VAE training on Cifar10 and ImageNet')
parser.set_defaults(lera=False)
parser.set_defaults(no_cuda=False)
parser.add_argument('--lera', dest='lera', action='store_true', help='Should i send training stats to https://learning-rates.com')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true', help='Run on CPU')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

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

datasets = {}

datasets['cifar10'] = torchvision.datasets.CIFAR10(root="./data/cifar10", download=True, train=True, transform=transforms.ToTensor())

imgnet_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

datasets['imagenet128'] = torchvision.datasets.ImageFolder(root="./data/imagenet/train", transform=imgnet_transform)

## Hyperparameters

batch_size = 128
lr = 2e-4
# categorical dimention
K = 64
# embedding size
D = 64
beta = 0.25 
total_steps = 250000

dataset = 'cifar10' 
#dataset = 'imagenet128'

field_size = 8 if dataset== 'cifar10' else 32

## TensorComprehension module to calculate distance between embedding and encoder output
# it saves a lot of memory

cross_dist_lang = """
def cross_dist(float(N,C,H,W) X, float(K, C) EMB) -> (dist) {
   dist(n, k, h, w) +=! pow(fabs(X(n, c, h, w) - EMB(k, c)), 2.0)
}
"""
cross_dist_cuda = tc.define(cross_dist_lang, name="cross_dist")

autotuned = dict()

def autotune(batch_size, K, D, field_size):
    global autotuned
    cache_autotune = ".autotune_{0:02d}_{1:02d}_{2:02d}_{3:02d}".format(batch_size, K, D, field_size)
    if autotuned.get(cache_autotune, None) is not None:
        return cache_autotune
    if not os.path.exists(cache_autotune + ".options") and use_cuda:
        input = torch.randn(batch_size, D, field_size, field_size).cuda()
        emb = torch.randn(K, D).cuda()
        cross_dist_cuda.autotune(input, emb , cache=cache_autotune, generations=10)
        autotuned[cache_autotune] = True
    return cache_autotune

def min_dist(input, emb):
    if use_cuda:
        cache_autotune = autotune(input.size(0), emb.size(0), emb.size(1), input.size(2))
        return cross_dist_cuda(input, emb, cache=cache_autotune).sub(V(sensitivity.view(1, K, 1, 1))).min(1)[1]

    # cpu version
    return (input.permute(0, 2, 3, 1) # [batch_size, w, h, D]
             .unsqueeze(-2)           # [batch_size, w, h, 1, D]
             .sub(emb)                # [batch_size, w, h, K, D]
             .pow(2).sum(-1)          # [batch_size, w, h, K]
             .sub(V(sensitivity.view(1, 1, 1, K)))
             .min(-1)[1])

emb_history = []

def train(epoch, step):
    #lera.log('epoch', epoch)
    epoch += 1

    for input, _ in DataLoader(datasets[dataset], batch_size=batch_size, pin_memory=use_cuda, num_workers=2, shuffle=True, drop_last=True):
        if use_cuda:
            input = input.cuda()

        step += 1

        ze = enc(V(input))

        index = min_dist(V(ze.data), embeddings)
        sz = index.size()

        zq = (embeddings[index.view(-1)]       # [batch_size * x * x, D] containing vectors from embeddings
                .view(sz[0], sz[1], sz[2], D)  # [batch_size, x, x, D] 
                .permute(0, 3, 1, 2))          # [batch_size, D, x, x]

        emb_loss = (zq - V(ze.data)).pow(2).sum(1).mean() + 1e-2 * embeddings.pow(2).mean()

        # detach zq so it won't backprop to embeddings with recon loss
        zq = V(zq.data, requires_grad=True)

        output = dec(zq)

        commit_loss = beta * (ze - V(zq.data)).pow(2).sum(1).mean()
        recon_loss = F.mse_loss(output, V(input))

        optimizer.zero_grad()

        commit_loss.backward(retain_graph=True)
        emb_loss.backward()
        recon_loss.backward()

        # pass data term gradient from decoder to encoder
        ze.backward(zq.grad)

        optimizer.step()

        emb_count[index.data.view(-1)] = 1
        emb_count.sub_(0.01).clamp_(min=0)
        unique_embeddings = emb_count.gt(0).sum()

        sensitivity.add_(emb_loss.data[0] * (K - unique_embeddings) / K)
        sensitivity[emb_count.gt(0)] = 0

        lera.log({ 
            'recon_loss': recon_loss.data[0],
            'commit_loss': commit_loss.data[0],
            'unique_embeddings': emb_count.gt(0).sum(),
            }, console=True)

        # make comparison image
        if lera.every(seconds=60):
            input = input.cpu()[0:8,:,:,:]
            w = input.size(-1)
            output = output.data.cpu()[0:8,:,:,:]
            result = (torch.stack([input, output])           # [2, 8, 3, w, w]
                        .transpose(0, 1).contiguous()        # [8, 2, 3, w, w]
                        .view(4, 4, 3, w, w)                 # [4, 4, 3, w, w]
                        .permute(0, 3, 1, 4, 2).contiguous() # [4, w, 4, w, 3]
                        .view(w * 4, w * 4, 3))              # [w * 4, w * 4, 3]
            lera.log_image('reconstruction', result.numpy(), clip=(0, 1))

    # continue training
    if step < total_steps:
        train(epoch, step)

## Train

lera.enabled(args.lera)
lera.log_hyperparams({
    'title' : "VQ-VAE",
    'dataset': dataset,
    'batch_size': batch_size,
    'K': K,
    'lr': lr,
    'D': D,
    'beta': beta,
    'total_steps' : total_steps
    })
lera.log_file(__file__)

enc = encoder(D) 
dec = decoder(D)
embeddings = torch.randn(K, D).div(D)
sensitivity = torch.zeros(K)

# calculate number of embedding vectors used
emb_count = torch.zeros(K)

if use_cuda:
    enc = enc.cuda()
    dec = dec.cuda()
    embeddings = embeddings.cuda()
    emb_count = emb_count.cuda()
    sensitivity = sensitivity.cuda()

embeddings = nn.Parameter(embeddings, requires_grad=True)
optimizer = optim.Adam([ 
    { 'params' : enc.parameters() },
    { 'params' : dec.parameters() },
    { 'params' : embeddings }
    ], lr=lr)

train(1, 1)
