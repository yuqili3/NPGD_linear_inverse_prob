import imageio
import os
import numpy as np
from PIL import Image

root = './data/celebA/testing_images/data'
save_root = './data/celebA/resized_testing_images/data'

if not os.path.isdir(save_root):
    os.makedirs(save_root)
img_list = os.listdir(root)

for i in range(len(img_list)):
    src = '%s/%s'%(root, img_list[i])
    im_orig = imageio.imread(src)
    h, w = im_orig.shape[:2]
    j = int(round((h - 108)/2.))
    k = int(round((w - 108)/2.))
    im = np.array(Image.fromarray(im_orig[j:j+108, k:k+108]).resize([64,64]))
    dst = '%s/%s'%(save_root,img_list[i])
    imageio.imwrite(dst, im)
        
#    if i%100==0:
#        print('%d/%d completed'%(i,len(img_list)))