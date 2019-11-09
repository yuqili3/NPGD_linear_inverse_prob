import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import argparse
import modelsMNIST as models
import os

parser = argparse.ArgumentParser(description='NPGD for mnist')
parser.add_argument('--datadir', default='./data/mnist', type=str, help='directory of mnist dataset')
parser.add_argument('--modeldir', default='./trained_model', type=str, help='directory of model training')
parser.add_argument('--resultdir', default='./result/mnist', type=str, help='directory of recovered image results')
parser.add_argument('--figuredir', default='./figure', type=str, help='directory of recovered image results')

parser.add_argument('--invGsigma', default=1,type=float, help='parameter in invG, level of noise added')
parser.add_argument('--invGlamda', default=0.1,type=float, help='parameter in invG, weight of multi-task loss')
parser.add_argument('--k', default=100, type=int, help='Latent variable dimension')

parser.add_argument('--task', default='cs', type=str, help='cs-compressed sensing, ip=inpainting, sr=superresolution')
parser.add_argument('--m', default=100, type=int, help='Measurement dimension in compressed sensing')
parser.add_argument('--designed', default=0, type=int, help='whether to use random matrices or designed matrices in CS')
parser.add_argument('--mask_size', default=7, type=int, help='number of pixels masked in inpainting')
parser.add_argument('--sr_ratio', default=2, type=int, help='ratio of super resolution in terms of side length')

parser.add_argument('--lr', default=1, type=float, help='Learning rate')
parser.add_argument('--ite', default=30, type=int, help='Number of NPGD iterations')
parser.add_argument('--info', default=0, type=int, help='whether to save intermediate result')


args = parser.parse_args()
data_dir = args.datadir
model_dir = args.modeldir
result_dir = args.resultdir
figure_dir = args.figuredir
invGsigma = args.invGsigma
invGlamda = args.invGlamda
if not os.path.isdir(figure_dir): os.makedirs(figure_dir)
if not os.path.isdir(result_dir): os.makedirs(result_dir)
k = args.k


task = args.task
use_designed_mat = args.designed > 0
m = args.m
mask_size = args.mask_size
sr_ratio = args.sr_ratio

batch_size = 12
lr=args.lr
num_ite= args.ite
save_intermediate = args.info > 0

def save_output(output_images, k, iterations, orig_paths, lr):
    path=save_path
    if task == 'cs':
        dir_name = '%s/%s_k%d_m%d_iter%d_lr%.2f' %(path,task_name, k, m, iterations, lr)
    elif task == 'sr':
        dir_name = '%s/%s_k%d_iter%d_lr%.2f_factor%d'%(path,task_name, k, iterations, lr, sr_ratio)
    elif task == 'ip':
        dir_name = '%s/%s_k%d_iter%d_lr%.2f_mask%d'%(path, task_name, k, iterations, lr, mask_size)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    output_images = output_images.detach().cpu().numpy()
    B = output_images.shape[0]
    for i in range(B):
        path = orig_paths[i]
        im = (output_images[i].transpose(1,2,0)+1)/2
        im = (im*255).astype(np.uint8)
        out_name = path.split('/')[-1]
        imageio.imwrite('%s/%s'%(dir_name,out_name),im)
        
def save_intermediate_cs(groundtruth, output_images, e, k,m):
    path = '%s/%s_%d_k%d_m%d.png'%(save_path, task_name, e,k,m)
    fig= plt.figure()
    groundtruth = groundtruth.detach().cpu().numpy()
    output_images = output_images.detach().cpu().numpy()
    B = groundtruth.shape[0]
    for i in range(B):
        plt.subplot(1,12,i+1)
        t,o = (groundtruth[i].transpose(1,2,0)+1)/2,(output_images[i].transpose(1,2,0)+1)/2
        combined = np.concatenate((t,o),axis=0).squeeze()
        plt.imshow(combined,cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(path)
    plt.close()
    

def save_intermediate_super(y, groundtruth, output_images, e, k, factor=sr_ratio):
    path = '%s/%s_%d_k%d_sr%d.png' % (save_path, task_name, e, k, factor)
    fig = plt.figure()
    groundtruth = groundtruth.detach().cpu().numpy()
    output_images = output_images.detach().cpu().numpy()
    y = y.reshape(-1, 1, int(28/factor), int(28/factor)).detach().cpu().numpy()
    y = np.kron(y, np.ones((factor, factor)))
    B = groundtruth.shape[0]
    for i in range(B):
        plt.subplot(1, 12, i + 1)
        # print(groundtruth[i].shape, y[i].shape)
        p, t, o = (y[i].transpose(1, 2, 0) + 1)/(2), (groundtruth[i].transpose(1, 2, 0) + 1) / 2, (output_images[i].transpose(1, 2, 0) + 1) / 2
        combined = np.concatenate((p, t, o), axis=0).squeeze()
        plt.imshow(combined, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(path)
    plt.close()


def save_intermediate_inpaint(y, groundtruth, output_images, e, k, mask_size=mask_size):
    path = '%s/%s_%d_k%d_mask%d.png' % (save_path, task_name, e, k, mask_size)
    fig = plt.figure()
    groundtruth = groundtruth.detach().cpu().numpy()
    output_images = output_images.detach().cpu().numpy()
    y = y.reshape(-1, 1, 28, 28).detach().cpu().numpy()
    B = groundtruth.shape[0]
    for i in range(B):
        plt.subplot(1, 12, i + 1)
        p, t, o = (y[i].transpose(1, 2, 0) + 1) / 2, (groundtruth[i].transpose(1, 2, 0) + 1) / 2, (output_images[i].transpose(1, 2, 0) + 1) / 2
        combined = np.concatenate((p, t, o), axis=0).squeeze()
        plt.imshow(combined, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(path)
    plt.close()

def get_super_resol_A(factor):
    A = np.zeros(shape=(int(28/factor)**2, 784))
    l = 0
    for i in range(int(28/factor)):
        for j in range(int(28/factor)):
            a = np.zeros(shape=(28, 28))
            a[factor*i:factor*(i+1), factor*j:factor*(j+1)] = 1/factor**2
            A[l, :] = np.reshape(a, [1, -1])
            l += 1
    return A

def get_inpaint_A(mask_size):
    A = np.ones((28, 28))
    A[int(28/2 - mask_size/2):int(28/2+mask_size/2), int(28/2 - mask_size/2):int(28/2+mask_size/2)] = 0
    A = A.ravel()
    A = np.diag(A)
    return A

gen_path = '%s/MNIST_DCGAN_results_%d/generator_param.pkl'%(model_dir, k)
G = models.DCGAN_generator(k=k, d=128)
G.load_state_dict(torch.load(gen_path))
G.cuda()

invG_path = '%s/MNIST_DCGAN_results_%d/invGen_param_mtloss_noisyGANoutput_sigma%.2f_lamda%.2f.pkl'%(model_dir,k,invGsigma, invGlamda);
invG = models.inverse_DCGAN(k=k, d=128)
invG.load_state_dict(torch.load(invG_path))
invG.cuda()

G.eval()
invG.eval()

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# testing MNIST dataset
data_dir = '%s/testing_images'%(data_dir)
class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index): 
        # this is what ImageFolder normally returns 
        original_tuple = super(MyImageFolder, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
dataset = MyImageFolder(data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Sensing matrix from user-defined arguments
if task == 'cs':
    task_name = 'compressed_sensing'
    if use_designed_mat:
        A = torch.load("%s/MNIST_DCGAN_results_%d/stiefelFromG_A_m%d.pkl"%(model_dir,k,m));
    else:
        # load learned A with ON rows, s.t. aligns best with G(z)
        A = np.random.normal(scale = np.sqrt(1/m), size=(m,1,28,28))
        A = torch.from_numpy(A).float();
elif task == 'ip':
    task_name = 'inpainting'
    # load A which applies mask (inpainting) with some mask size
    A = get_inpaint_A(mask_size); 
    m = A.shape[0]
    A = A.reshape(784, 1, 28, 28)
    A = torch.from_numpy(A)
elif task == 'sr':
    task_name = 'super_resolution'
    # load A which does super-resolution with sr factor
    A = get_super_resol_A(sr_ratio); 
    m = A.shape[0]
    A = A.reshape(m, 1, 28, 28)
    A = torch.from_numpy(A)
# create results directory
save_path='%s/%s'%(result_dir, task_name)
if not os.path.isdir(save_path):
    os.makedirs(save_path)

# Perform A@x in torch, except now x: Bx1x28x28, A: mx1x28x28, return shape Bxm
def Aat(x):
    y = (A*x.unsqueeze(1)).sum(dim=(2,3,4))
    return y
# Perform A.T@y in torch, except now y: Bxm, A: mx1x28x28, return shape mx1x28x28
def ATat(y):
    x = (A.permute(1,2,3,0).unsqueeze(3)) * y # now is 3x64x64xBxm
    x = x.sum(dim=4) # now is 3x64x64xB
    x = x.permute(3,0,1,2)
    return x
    
# perform NPGD 
for x_, _, paths in testloader:
    x_, A  = x_.cuda(), A.cuda().float() # now x_: Bx3x38x38
    x_ = x_[:,0,:,:].unsqueeze(dim=1) # now x_: Bx1x28x28
    
    y = Aat(x_)
    x_t = x_init = ATat(y); 
    
    for ii in range(num_ite):
        grad_term = ATat(Aat(x_t) - y)
        z_t = x_t - lr*grad_term
        x_t = G(invG(z_t))
        print('Iteration %d, ||y-Axt||_2: %.4f, ||x*-xt||_2: %.4f'%(ii, torch.norm(y - Aat(x_t)), torch.norm(x_t - x_)) )
        if save_intermediate: # save intermediate output
            if task == 'cs': save_intermediate_cs(x_, x_t, ii, k, m)
            elif task == 'sr': save_intermediate_super(y, x_, x_t, ii, k, factor=sr_ratio)
            elif task == 'ip': save_intermediate_inpaint(y, x_, x_t, ii, k, mask_size=mask_size)
    save_output(x_t, k,num_ite,paths,lr) 
    if save_intermediate:
        break # save one batch only
        
if save_intermediate:
    images = []
    for e in range(0,num_ite):
        if task == 'cs': img_name = '%s/%s_%d_k%d_m%d.png'%(save_path, task_name, e,k,m)
        elif task == 'ip': img_name = '%s/%s_%d_k%d_mask%d.png' % (save_path, task_name, e, k, mask_size)
        elif task == 'sr': img_name = '%s/%s_%d_k%d_sr%d.png' % (save_path, task_name, e, k, sr_ratio)
        images.append(imageio.imread(img_name))
    imageio.mimsave('%s/mnist_%s_example.gif'%(figure_dir, task_name), images, fps=5)
        
   

