import numpy as np
import numpy.linalg as la
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os.path
import argparse

parser = argparse.ArgumentParser(description='choose measurement matrix A')
parser.add_argument('--dataset', default='CELEBA', type=str, help='MNIST or CELEBA')
parser.add_argument('--modeldir', default='./trained_model', type=str, help='directory of model training')
parser.add_argument('--k', default=100, type=int, help='Latent variable dimension')

args = parser.parse_args()
dataset = args.dataset
model_dir = args.modeldir
k = args.k

if dataset == 'MNIST':
    import modelsMNIST as models
    img_size = 784
    mlist = [10,20,30,50,100,200,500]
elif dataset == 'CELEBA':
    import modelsCelebA as models
    img_size = 64*64*3
    mlist = [100,200,500,1000,2000,5000]
    

gen_path = '%s/%s_DCGAN_results_%d/generator_param.pkl'%(model_dir, dataset, k)
G = models.DCGAN_generator(k=k, d=128)
G.load_state_dict(torch.load(gen_path))
G.cuda() 

out = '%s/%s_DCGAN_results_%d/DDT_from_Gz.npz'%(model_dir,dataset, k)
if os.path.isfile(out):
    db = np.load(out)
    U, L = db['U'], db['L']
else:
    G.eval()
    if dataset=='MNIST':
        N = 10000 # number of x_i = G(z) generated in tatal
        M = 50000 # number of random pair xi, xj in total
    elif dataset=='CELEBA':
        N = 20000
        M = 100000
        
    bN = 500; bM = M*bN//N 
    # generate bN images and choose bM pairs from them, to fit into memory, no money no gpu
    # U can adjust bN to fit it into GPU
    batch_size=500 
    DDT = torch.zeros(img_size, img_size).cuda()
    start_time = time.time()
    with torch.no_grad():
        for iter1 in range(N//bN):
            s0 = time.time()
            x = torch.randn(bN,k,1,1).cuda()
            x = G(x).view(bN,img_size)
            e0 = time.time()
            print('e0-s0', e0-s0)
        
            randidx = np.random.randint(bN**2, size=bM)
            I,J = randidx//bN, randidx%bN
        
            DT = torch.zeros(batch_size, img_size).cuda()
            s1 = time.time()
            for iter2 in range(bM // batch_size):
                DT.fill_(0)
                for t in range(batch_size):
                    i,j=I[t],J[t]
                    d_norm = torch.norm(x[i]-x[j])
                    if d_norm > 1e-5:
                        DT[t] = (x[i]-x[j])/d_norm
            e1 = time.time()
            print('e1-s1', e1-s1)
            s2 = time.time()
            DDT += sum(torch.ger(DT[i],DT[i])/N for i in range(batch_size))
            e2 = time.time()
            print('e2-s2',e2-s2)
            print('%d/%d'%(iter1, N//bN))
        DDT = DDT.detach().cpu().numpy()
    end_time = time.time()
    print('construct DDT done, time elapsed: ', end_time-start_time)
    
    s3=time.time()
    L, U = la.eig(DDT)
    e3 =time.time()
    print('eigendecomp DDT done, time elapsed: ', e3-s3)
    idx = np.real(L).argsort()[::-1] 
    L,U = L[idx], U[:,idx] # sort eigen values from highest to lowest
    out = '%s/%s_DCGAN_results_%d/DDT_from_Gz'%(model_dir,dataset, k)
    np.savez(out, DDT=DDT,L=L, U=U)

#    m=500
#    A = np.real(U[:,:m]).T
#    ATA = A.T@A
#    print(ATA[:5,:5])
#    exit()
for m in mlist: 
#        print('eigenvalues of DDT, ',np.real(L[:m]))
    if dataset == 'MNIST':  
        A = torch.from_numpy(np.real(U[:,:m]).T.reshape(m,1,28,28)).float().cuda()
    elif dataset =='CELEBA':
        A = torch.from_numpy(np.real(U[:,:m]).T.reshape(m,3,64,64)).float().cuda()
    torch.save(A, "%s/%s_DCGAN_results_%d/designed_matrix_A_m%d.pkl"%(model_dir,dataset,k,m))
    print("saving matrix A...")




