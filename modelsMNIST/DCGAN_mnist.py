import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class DCGAN_generator(nn.Module):
    # initializers
    def __init__(self,k=100, d=128):
        super(DCGAN_generator, self).__init__()
        # maybe try kernel=5, stride=1/2, pad=1?
        self.deconv1 = nn.ConvTranspose2d(k, d*4, 3, 1, 0) # 1x1->3x3
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 3, 2, 0) # 3x3->7x7
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) # 7x7->14x14
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1) # 14x14->28x28

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(input)),0.2)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)),0.2)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)),0.2)
        x = F.dropout(x, 0.5)
        x = torch.tanh(self.deconv4(x))
        return x

class DCGAN_discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(DCGAN_discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1) # 28x28->14x14
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1) #14x14->7x7
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 2, 0) # 7x7->3x3 
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, 1, 3, 1, 0) # 3x3->1x1

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.dropout(x, 0.5)
        x = torch.sigmoid(self.conv4(x))

        return x

class inverse_DCGAN(nn.Module):
    # input is Bx1x28x28 image, output is Bxkx1x1
    def __init__(self, k=100,d=128):
        self.d,self.k = d,k
        super(inverse_DCGAN,self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1) # 28x28->14x14
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1) #14x14->7x7
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 2, 0) # 7x7->3x3 
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*4, 3, 1, 0) # 3x3->1x1
        self.fc = nn.Linear(d*4, k)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.dropout(x, 0.5)
        x = self.conv4(x)
        x = x.view(-1, self.d*4)
        x = self.fc(x)
        x = x.view(-1,self.k, 1, 1)
        return x 


if __name__=='__main__':
    invG = inverse_DCGAN(k=100)
    G = DCGAN_generator(k=100)
    num = 2
    x = torch.randn(num,1,28,28)
    invG_x = invG(x)
    x_proj = G(invG_x)
    print(x_proj.size())
#    





        