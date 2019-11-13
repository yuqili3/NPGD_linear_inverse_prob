# NPGD_linear_inverse_prob
This repository provides code for this paper: [GAN-based Projector for Faster Recovery with Convergence Guarantees in Linear Inverse Problems](https://arxiv.org/abs/1902.09698). 

## Requirements
Linux system, a GPU 
1. Python 3.4
2. Pytorch 1.2.0
3. numpy  
4. matplotlib 
5. imageio 
6. scikit-image 
7. scikit-learn 
8. requests 
9. torchvision 



## Preliminaries
1. Create a conda virtual environment and install packages
```
conda create --name NPGD python=3.4
conda activate NPGD
conda install pytorch=1.2.0 imageio=2.6.1 matplotlib=3.1.1 numpy=1.17.2 scikit-image=0.15.0 scikit-learn=0.21.3 requests=2.22.0 torchvision=0.4.0 
```

2. Clone this repo 
```
git clone https://github.com/yuqili3/NPGD_linear_inverse_prob.git
cd NPGD_linear_inverse_prob
```

3. Download the dataset as well as the test images, we need to crop and resize the faces from the original images. 
```
unzip celebAtest.zip -d data/celebA/testing_images
unzip mnisttest.zip -d data/mnist/
python download_data.py celebA
rm -rf ./data/celebA/celebA_data_raw
rm ./data/celebA/celebA_data.zip
```
this may take several minutes to download the large file

4. Train the network projector models.
    1. for mnist dataset, train a DCGAN first and then a concatenated network projector:
    ```
    python MNIST_DCGAN.py --k=100 --epochs=40 --lr=1e-4
    python MNIST_invDCGAN_multi_task_loss.py --k=100 --epochs=100 --lr=1e-4 --lamda=0.1 --sigma=1
    ```
    2. for celebA dataset:
    ```
    python CELEBA_DCGAN.py --k=100 --epochs=40 --lr=1.5e-4
    python CELEBA_invDCGAN_multi_task_loss.py --k=100 --epochs=100 --lr=1e-4 --lamda=0.1 --sigma=1
    ```
    
    
## Demos
1. Compressed sensing using random Gaussian matrix
    * mnist
    ```
    python MNIST_projection_using_inv_gen.py --k=100 --task=cs --m=100 --designed=0  --lr=1 --ite=30 --info=1
    ```
    * celebA
    ```
    python CELEBA_projection_using_inv_gen.py --k=100 --task=cs --m=1000 --designed=0  --lr=0.5 --ite=30 --info=1
    ```
    
2. Image inpainting
    * mnist
    ```
    python MNIST_projection_using_inv_gen.py --k=100 --task=ip --mask_size=8 --lr=2 --ite=30 --info=1
    ```
    * celebA
    ```
    python CELEBA_projection_using_inv_gen.py --k=100 --task=ip --mask_size=32 --lr=1 --ite=30 --info=1
    ```
    
3. Image super resolution
    * mnist
    ```
    python MNIST_projection_using_inv_gen.py --k=100 --task=sr --sr_ratio=2 --lr=6 --ite=30 --info=1
    ```
    * celebA
    ```
    python CELEBA_projection_using_inv_gen.py --k=100 --task=sr --sr_ratio=2 --lr=6 --ite=30 --info=1
    ```

4. Use a designed matrix in compressed sensing

   * mnist
   ```
   python design_measurement_matrix.py --dataset=MNIST --k=100
   python MNIST_projection_using_inv_gen.py --k=100 --task=cs --m=200 --designed=1 --lr=1 --ite=30 --info=1
   ```
   
   * celebA
   ```
   python design_measurement_matrix.py --dataset=CELEBA --k=100
   python CELEBA_projection_using_inv_gen.py --k=100 --task=cs --m=1000 --designed=1 --lr=1 --ite=30 --info=1
   ```
