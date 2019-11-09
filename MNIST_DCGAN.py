import os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import modelsMNIST as models
import argparse

parser = argparse.ArgumentParser(description='GAN with diff k')
parser.add_argument('--datadir', default='./data/mnist', type=str, help='directory of mnist dataset')
parser.add_argument('--modeldir', default='./trained_model', type=str, help='directory of model training')
parser.add_argument('--k', default=100, type=int, help='Latent variable dimension')
parser.add_argument('--epochs', default=40, type=int, help='Total learning epoch')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
args = parser.parse_args()
data_dir = args.datadir
model_dir = args.modeldir
k = args.k
lr = args.lr
epochs = args.epochs


fixed_z_ = torch.randn((5 * 5, k)).view(-1, k, 1, 1)    # fixed noise
with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.cuda())

def show_result(num_epoch,k=100, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, k)).view(-1, k, 1, 1)
    with torch.no_grad():
        z_ = Variable(z_.cuda())

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 128
lr = lr
train_epoch = epochs


# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
]) # image value => [-1,1]
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = models.DCGAN_generator(k=k)
D = models.DCGAN_discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss for opriginal gan
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
dir_name = '%s/MNIST_DCGAN_results_%d' % (model_dir, k)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
if not os.path.isdir('%s/Random_results' % dir_name):
    os.mkdir('%s/Random_results' % dir_name)
if not os.path.isdir('%s/Fixed_results' % dir_name):
    os.mkdir('%s/Fixed_results' % dir_name)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for x_, y_ in train_loader:
        batch_size = x_.size()[0]
        y_real_, y_fake_ = torch.ones(batch_size), torch.zeros(batch_size)
        
        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        
        # train discriminator D. 
        D.zero_grad()
        D_real_result = D(x_).squeeze()

        z_ = torch.randn((batch_size, k)).view(-1, k, 1, 1)
        z_ = Variable(z_.cuda())
        D_fake_result = D(G(z_)).squeeze()
        D_fake_score = D_fake_result.data.mean()

        D_train_loss =  BCE_loss(D_real_result, y_real_) + BCE_loss(D_fake_result, y_fake_)

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        z_ = torch.randn((batch_size, k)).view(-1, k, 1, 1)
        z_ = Variable(z_.cuda())

        D_result = D(G(z_)).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.item())


    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = '%s/Random_results/MNIST_DCGAN_%d.png'%(dir_name,epoch + 1)
    fixed_p = '%s/Fixed_results/MNIST_DCGAN_%d.png'%(dir_name,epoch + 1)
    show_result((epoch+1), k=k, save=True, path=p, isFix=False)
    show_result((epoch+1), k=k, save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "%s/generator_param.pkl"%(dir_name))
torch.save(D.state_dict(), "%s/discriminator_param.pkl"%(dir_name))
with open('%s/train_hist.pkl'%(dir_name), 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='%s/MNIST_DCGAN_train_hist.png'%(dir_name))

images = []
for e in range(train_epoch):
    img_name = '%s/Fixed_results/MNIST_DCGAN_%d.png'%(dir_name,e + 1)
    images.append(imageio.imread(img_name))
imageio.mimsave('%s/generation_animation.gif'%(dir_name), images, fps=5)