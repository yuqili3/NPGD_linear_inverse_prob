import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import normal
from torch.autograd import Variable
import torch.autograd as autograd
import modelsV1_2 as models

    
'''
case 1: if \delta = 0
gamma_min := min_{z1,z2, ||z1-z2||} > \epsilon} ||AG(z1)-AG(z2)||^2 / ||G(z1)-G(z2)||^2
using alternating min, alternate between z1 and z2
'''

# Perform A@x in torch, except now x: Bx1x28x28, A: mx28x28, return shape Bxm
def Aat(x):
    y = (A*x).sum(dim=(2,3))
    return y


def REC(A, k=100, lr=1e-1,T=200, T_in=5):
    gen_path = '/home/yuqi/Documents/Research/inv_gan_no_constraint_on_z/train_without_discriminator/L2_loss/MNIST_DCGAN_results_%d/generator_param.pkl'%(k)
    G = models.DCGAN_generator(k=k, d=128)
    G.load_state_dict(torch.load(gen_path))
    G.cuda()

#    A = torch.from_numpy(A).float().cuda()
    A = A.cuda()
    
    z1 = normal.Normal(loc=0, scale=1).sample((1, k)).view(-1, k, 1, 1) 
    z2 = z1 + 1e-2*normal.Normal(loc=0,scale=1).sample((1, k)).view(-1, k, 1, 1)
    z1,z2 = Variable(z1).cuda(), Variable(z2).cuda()
    z1.requires_grad_(True)
    z2.requires_grad_(True)
    
    gamma_min=torch.Tensor([1]).float().cuda()
    for t in range(T):
        x1 = G(z1)
        Ax1 = Aat(x1)
        for t_in in range(T_in):
            output = (torch.norm(Ax1 - Aat(G(z2))) / torch.norm(x1 - G(z2)))**2
            gradients = autograd.grad(outputs=output, inputs=z2,
                                              grad_outputs=torch.ones(output.size()).cuda(),
                                              create_graph=False, retain_graph=False, only_inputs=True)[0]
            z2 = z2 - lr*gradients
            gamma_min = torch.min(output,gamma_min)
                
            
        x2 = G(z2)
        Ax2 = Aat(x2)
        for t_in in range(T_in):
            output = ( torch.norm(Ax2 - Aat(G(z1))) / torch.norm(x2 - G(z1)))**2
            gradients = autograd.grad(outputs=output, inputs=z1,
                                              grad_outputs=torch.ones(output.size()).cuda(),
                                              create_graph=False, retain_graph=False, only_inputs=True)[0]
            z1 = z1 - lr*gradients
            gamma_min = torch.min(output,gamma_min)
#        print('smallest gamma observed: ',gamma_min)
    
    z1 = normal.Normal(loc=0, scale=1).sample((1, k)).view(-1, k, 1, 1)
    z2 = z1 + 1e-2*normal.Normal(loc=0,scale=1).sample((1, k)).view(-1, k, 1, 1)
    z1,z2 = Variable(z1).cuda(), Variable(z2).cuda()
    z1.requires_grad_(True)
    z2.requires_grad_(True)   
    gamma_max=torch.Tensor([1]).float().cuda()
    
    for t in range(T):
        x1 = G(z1)
        Ax1 = Aat(x1)
        for t_in in range(T_in):
            output = (torch.norm(Ax1 - Aat(G(z2))) / torch.norm(x1 - G(z2)))**2
            gradients = autograd.grad(outputs=output, inputs=z2,
                                              grad_outputs=torch.ones(output.size()).cuda(),
                                              create_graph=False, retain_graph=False, only_inputs=True)[0]
            z2 = z2 + lr*gradients
            gamma_max = torch.max(output,gamma_max)
        x2 = G(z2)
        Ax2 = Aat(x2)
        for t_in in range(T_in):
            output = ( torch.norm(Ax2 - Aat(G(z1))) / torch.norm(x2 - G(z1)))**2
            gradients = autograd.grad(outputs=output, inputs=z1,
                                              grad_outputs=torch.ones(output.size()).cuda(),
                                              create_graph=False, retain_graph=False, only_inputs=True)[0]
            z1 = z1 + lr*gradients
            gamma_max = torch.max(output,gamma_max)
#        print('largest gamma observed: ',gamma_max)
    
    print('smallest gamma observed: ',gamma_min)
    print('largest gamma observed: ',gamma_max)
    
#    U,D,V = np.linalg.svd(A)
#    print('||A||^2',max(D)**2) 
#    print('\sigma_min^2',min(D)**2)
    print('distance of z1,z2: ',torch.norm(z1-z2))
    print('distance of x1,x2: ',torch.norm(G(z1)-G(z2)))
    return gamma_min, gamma_max

if __name__=='__main__':
#    m_list = [200,300,400,500]
#    k_list = [10,20,30,50,100]
    root_path = '/home/yuqi/Documents/Research/inv_gan_no_constraint_on_z/train_without_discriminator/L2_loss'
    m_list = [200] # 30,50,100,
    k_list = [100] #[20,30,50,100]
    for m in m_list:
        for k in k_list:
            print('k',k,'m',m)
            num_copy = 1
            mi, ma = np.zeros(num_copy),np.zeros(num_copy)
            for copy in range(num_copy):
#                A = np.random.normal(loc=0, scale=1/np.sqrt(m), size=(m, 28*28)) # common normal
#                np.random.seed(0)
#                A = np.sqrt(1/m)* np.random.randn(m,28*28)
                A = torch.load("%s/MNIST_DCGAN_results_%d/stiefelFromG_A_combined_m%d.pkl"%(root_path,k,m))
                mi[copy],ma[copy] = REC(A,k=k)
            np.savez('REC_param_est/REC_param_stiefelA_m%d_k%d_tmp'%(m,k),mi=mi,ma=ma)
    
    '''
    k = 100
    m_list = [100,200,300,400,500,600,700]
    for m in m_list:
        num_copy=100
        mi, ma = np.zeros(num_copy),np.zeros(num_copy)
        for copy in range(num_copy):
            A = np.random.normal(loc=0, scale=1/np.sqrt(m), size=(m, 28*28)) # common normal
            mi[copy],ma[copy] = REC(A)
        np.savez('REC_param_est/REC_param_m%d_k%d'%(m,k),mi=mi,ma=ma)
    '''      

    