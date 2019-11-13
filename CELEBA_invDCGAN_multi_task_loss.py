import numpy as np
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import modelsCelebA as models
import os

# Parsing
parser = argparse.ArgumentParser(description='invGAN with diff k')
parser.add_argument('--datadir', default='./data/mnist', type=str, help='directory of celebA dataset')
parser.add_argument('--modeldir', default='./trained_model', type=str, help='directory of model training')
parser.add_argument('--k', default=100, type=int, help='Latent variable dimension')
parser.add_argument('--lr', default=1.5e-4, type=float, help='Learning rate')
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
parser.add_argument('--lamda', default=0.1, type=float, help='MTL regularizer')
parser.add_argument('--sigma', default=1, type=float, help='Noise Perturbation for Projector')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
data_dir = args.datadir
model_dir = args.modeldir
k = args.k
lr = args.lr
epochs = args.epochs
weight_latent_loss = args.lamda
sigma = args.sigma

def show_train_hist(hist, save = False, path = 'Train_hist.png'):
    x = range(len(hist['MSE_losses']))
    y1 = hist['MSE_losses']
    plt.plot(x, y1)
    plt.xlabel('Iter')
    plt.ylabel('MSE_Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)

def show_result(G, invG, path, nums=3):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    data_dir = './data/celebA/testing_images'
    dset = datasets.ImageFolder(data_dir, transform)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=nums, shuffle=False)
    plt.figure(figsize=(6,4))
    for x_batch,y in train_loader:
        for i in range(nums):
            x = x_batch[i].unsqueeze(0).cuda()
            x_proj = G(invG(x))
            plt.subplot(nums,2, 2*i+1)
            plt.imshow((x.detach().cpu().numpy().squeeze().transpose(1, 2, 0) + 1) / 2)
            plt.xticks([]);plt.yticks([])
            plt.title('CELEBA image x')
            plt.subplot(nums,2, 2*i+2)
            plt.imshow((x_proj.detach().cpu().numpy().squeeze().transpose(1, 2, 0) + 1) / 2)
            plt.xticks([]);plt.yticks([])
            plt.title('projection of x onto R(G)')
        break
    plt.savefig(path)
    plt.close()

# Dataloader from G(z)+noise
batch_size = 128
num_train_samples = 100000

# Forward Generator Load pretrained model
model_folder = '%s/CELEBA_DCGAN_results_%d'%(model_dir, k)
gen_path = '%s/generator_param.pkl'%(model_folder)
G = models.DCGAN_generator(k=k, d=128)
G.load_state_dict(torch.load(gen_path))
G.cuda()

# Inverse Generator
invG = models.inverse_DCGAN(k=k,d=128)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    invG_path = '%s/invGen_param_mtl_loss_noisyGANoutput_sigma%.2f_lamda%.2f.pkl'%(model_folder, sigma, weight_latent_loss)
    assert os.path.isfile(os.path.normpath(invG_path)), 'Error: %s found!'%(invG_path)
    invG.load_state_dict(torch.load(invG_path))    
invG.cuda()

# Training
print('Training Started')
learning_rate = lr
optimizer_invG = optim.Adam(invG.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
num_epochs = epochs

invG.train()
G.eval()
for p in G.parameters():
    p.requires_grad_(False)

train_hist = collections.defaultdict(list)
torch.manual_seed(0)
torch.cuda.manual_seed_all(1)

for epoch in range(num_epochs):
    avg_loss = []
    epoch_start_time = time.time()
    for i in range(int(num_train_samples/batch_size)):
        z = torch.randn(batch_size, k, 1, 1).cuda()
        x = G(z)
        xn = x + sigma*torch.randn(x.shape).cuda()
        invG.zero_grad()
        invG_out = invG(xn)
        G_out = G(invG_out)
        loss1 = criterion(G_out, x)
        loss2 = criterion(invG_out, z)
        loss = loss1 + weight_latent_loss*loss2
        loss.backward()
        optimizer_invG.step()
        avg_loss.append(loss.item())
        
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f, loss: %.4f '%(epoch+1,num_epochs,per_epoch_ptime, np.mean(avg_loss)))
    train_hist['MSE_losses'].append(np.mean(loss.item()))
    
    torch.save(invG.state_dict(), "%s/invGen_param_mtl_loss_noisyGANoutput_sigma%.2f_lamda%.2f.pkl" % (model_folder,sigma, weight_latent_loss))
    print("saving...")
    
with open('%s/invGen_train_hist_mtl_loss_noisyGANoutput.pkl' % (model_folder), 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path='%s/CELEBA_invDCGAN_train_hist_mtl_loss_noisyGANoutput.png' % (model_folder))
show_result(G, invG, path='%s/CELEBA_projection_examples_mtl_loss_noisyGANoutput.png'%(model_folder),nums=3)

