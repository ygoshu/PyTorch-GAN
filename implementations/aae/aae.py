import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision import utils

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys
sys.path.insert(1, '/home/yag3/WGAN/pytorch-wgan')
from utils.fashion_mnist import MNIST

os.makedirs("few_images", exist_ok=True)
os.makedirs("aae_results", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--test_batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

device = torch.device("cuda" if cuda else "cpu")

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)


trans = transforms.Compose([
    transforms.Scale(28),
    transforms.ToTensor(),
    transforms.Normalize(((0.5,0.5,0.5)), (0.5,0.5,0.5)),
])

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/train_mnist', train=True, download=True,
                   transform=trans),
    batch_size=opt.batch_size, shuffle=True, **kwargs)

test_emnist_dataset = MNIST(root='./data/test_emnist', train=False, download=True, transform=trans, few_shot_class=5, test_emnist=True)
test_emnist_loader  = torch.utils.data.DataLoader(test_emnist_dataset ,  batch_size=opt.test_batch_size, shuffle=True)

test_mnist_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/test_mnist', train=False, download=True, transform=trans),
    batch_size=opt.test_batch_size, shuffle=True, **kwargs)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, "few_images/%d.png" % batches_done, nrow=n_row, normalize=True)

def test_recon(epoch, test_loader):
    encoder.load_state_dict(torch.load('encoder_aae.pkl'))
    decoder.load_state_dict(torch.load('decoder_aae.pkl'))
    encoder.eval()
    decoder.eval()  
    discriminator.eval() 
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
           data = data.to(device)
           real_imgs = Variable(data.type(Tensor))
           encoded_imgs = encoder(real_imgs)
           decoded_imgs = decoder(encoded_imgs) 
           if i == 0:
               n = min(data.size(0), 8)
               comparison = torch.cat([data[:n],
                                     decoded_imgs.view(opt.test_batch_size, 1, 28, 28)[:n]])
               save_image(comparison.cpu(),
                        'aae_results/reconstruction_' + str(epoch) + '_' + str(is_emnist)  +'.png', nrow=n)
               break

def optimizeZ(test_loader):
       if not os.path.exists('gen_aae_res/'):
           os.makedirs('gen_aae_res/')
       encoder.load_state_dict(torch.load('encoder_aae.pkl'))
       decoder.load_state_dict(torch.load('decoder_aae.pkl'))
       encoder.eval()
       decoder.eval()
       discriminator.eval()
       #z = torch.randn(opt.batch_size, (opt.latent_dim)).to('cuda')
       z = Variable(Tensor(np.random.normal(0, 1, (opt.test_batch_size, opt.latent_dim))))
       z.requires_grad = True
       print("Checking if z requires Gradient")
       print(z.requires_grad)
       learning_rate = 0.0002
       opt_iter = 0

       loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
       print("self.epochs")
       epochs = 4000
       images = None
       for i, (image, _) in enumerate(test_loader):
           images = Variable(image.to(device).type(Tensor))
           grid_org = utils.make_grid(images)
           utils.save_image(grid_org, 'gen_aae_res/orig_ae_res_{}_{}.png'.format(str(opt_iter).zfill(3), str(is_emnist)))
           break
       optimizer = torch.optim.Adam([z], lr=learning_rate)
       grid = None
       og = None
       for epoch in range(epochs):
             #generate image
             x_recon = decoder(z)

             if opt_iter == 0:
                 og_recon = x_recon

             #calculate reconstruction loss 
             loss = loss_fn(x_recon, images)  #test_loader andk get first img)

             #zero out gradient so that the previous calculated gradient doesn't add on to the current calculated grad
             optimizer.zero_grad()

             #calculate gradient
             loss.backward()

             #update scale 
             optimizer.step()

             if opt_iter % 100 == 0:
                 print("Iter {}, loss {}".format(str(opt_iter), str(loss.item())))
                 #x_recon  = x_recon.mul(0.5).add(0.5)
                 #x_recon = x_recon.data.cpu()[:64]
                # grid = utils.make_grid(x_recon)
                 utils.save_image(x_recon, 'gen_aae_res/gen_ae_res_{}_{}.png'.format(str(opt_iter).zfill(3), str(is_emnist) ), nrow=opt.test_batch_size)

             opt_iter += 1
       comparison = torch.cat([og_recon, images, x_recon])  
       utils.save_image(comparison.cpu() , 'gen_aae_res/comparison_{}.png'.format(str(opt_iter).zfill(3), str(is_emnist)), nrow=opt.test_batch_size )

for epoch in range(1):
    is_emnist = 0 
    #test_recon(epoch=epoch,test_loader=test_mnist_loader)
    optimizeZ(test_loader=test_mnist_loader)
    is_emnist = 1
    optimizeZ(test_loader=test_emnist_loader)
    #test_recon(epoch=epoch, test_loader=test_emnist_loader)
exit(0)
# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
        torch.save(decoder.state_dict(), './decoder_aae.pkl')
        torch.save(encoder.state_dict(), './encoder_aae.pkl')
        torch.save(discriminator.state_dict(), './discriminator_aae.pkl')

for epoch in range(5):
    is_emnist = 0 
    test_recon(epoch=epoch,test_loader=test_mnist_loader)
    is_emnist = 1
    test_recon(epoch=epoch, test_loader=test_emnist_loader)
 
