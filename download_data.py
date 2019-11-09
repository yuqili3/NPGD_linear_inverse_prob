from __future__ import print_function
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import subprocess
from six.moves import urllib
#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
import imageio
from PIL import Image
import numpy

parser = argparse.ArgumentParser(description='Download datasets.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['celebA', 'lsun', 'mnist'],
           help='name of dataset to download [celebA, lsun, mnist]')

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
                
def download(url, dirpath):
  filename = url.split('/')[-1]
  filepath = os.path.join(dirpath, filename)
  u = urllib.request.urlopen(url)
  f = open(filepath, 'wb')
  filesize = int(u.headers["Content-Length"])
  print("Downloading: %s Bytes: %s" % (filename, filesize))

  downloaded = 0
  block_sz = 8192
  status_width = 70
  while True:
    buf = u.read(block_sz)
    if not buf:
      print('')
      break
    else:
      print('', end='\r')
    downloaded += len(buf)
    f.write(buf)
    status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
      ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
    print(status, end='')
    sys.stdout.flush()
  f.close()
  return filepath

def unzip(filepath):
  print("Extracting: " + filepath)
  dirpath = os.path.dirname(filepath)
  with zipfile.ZipFile(filepath) as zf:
    zf.extractall(dirpath)
  os.remove(filepath)

def download_celeb_a(dirpath):
  data_dir = 'celebA'
  if os.path.exists(os.path.join(dirpath, data_dir)):
    print('Found Celeb-A - skip')
    return
  else:
      os.mkdir(os.path.join(dirpath, data_dir))
  fileID = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
  download_file_from_google_drive(fileID, dirpath+'/celebA/celebA_data.zip')
  
  save_dir = './data/celebA/celebA_data_raw'
  if not os.path.isdir(save_dir):
      os.mkdir(save_dir)
  with zipfile.ZipFile('./data/celebA/celebA_data.zip','r') as zf:
    zf.extractall(save_dir)
  
    
def preprocess_celeb_a(dirpath):
    raw_dir = os.path.join(dirpath, 'celebA/celebA_data_raw/img_align_celeba')
    crop_dir = os.path.join(dirpath, 'celebA/celebA_data')
    if not os.path.isdir(crop_dir):
        os.mkdir(crop_dir)
    img_list = os.listdir(raw_dir)
    if not os.path.isdir('%s/train/data'%(crop_dir)):
        os.makedirs('%s/train/data'%(crop_dir))
    if not os.path.isdir('%s/test/data'%(crop_dir)):
        os.makedirs('%s/test/data'%(crop_dir))
    for i in range(len(img_list)):
        src = '%s/%s'%(raw_dir, img_list[i])
        im_orig = imageio.imread(src)
        h, w = im_orig.shape[:2]
        j = int(round((h - 108)/2.))
        k = int(round((w - 108)/2.))
        im = numpy.array(Image.fromarray(im_orig[j:j+108, k:k+108]).resize([64,64]))
        if i <= 160e3:
            dst = '%s/train/data/%s'%(crop_dir, img_list[i])
        else:
            dst = '%s/test/data/%s'%(crop_dir, img_list[i])
        imageio.imwrite(dst, im)
        if i%100==0:
            print('%d/%d completed'%(i, len(img_list)))
    
def _list_categories(tag):
  url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
  f = urllib.request.urlopen(url)
  return json.loads(f.read())

def _download_lsun(out_dir, category, set_name, tag):
  url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
      '&category={category}&set={set_name}'.format(**locals())
  print(url)
  if set_name == 'test':
    out_name = 'test_lmdb.zip'
  else:
    out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
  out_path = os.path.join(out_dir, out_name)
  cmd = ['curl', url, '-o', out_path]
  print('Downloading', category, set_name, 'set')
  subprocess.call(cmd)

def download_lsun(dirpath):
  data_dir = os.path.join(dirpath, 'lsun')
  if os.path.exists(data_dir):
    print('Found LSUN - skip')
    return
  else:
    os.mkdir(data_dir)

  tag = 'latest'
  #categories = _list_categories(tag)
  categories = ['bedroom']

  for category in categories:
    _download_lsun(data_dir, category, 'train', tag)
    _download_lsun(data_dir, category, 'val', tag)
  _download_lsun(data_dir, '', 'test', tag)

def download_mnist(dirpath):
  data_dir = os.path.join(dirpath, 'mnist')
  if os.path.exists(data_dir):
    print('Found MNIST - skip')
    return
  else:
    os.mkdir(data_dir)
  url_base = 'http://yann.lecun.com/exdb/mnist/'
  file_names = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
  for file_name in file_names:
    url = (url_base+file_name).format(**locals())
    print(url)
    out_path = os.path.join(data_dir,file_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading ', file_name)
    subprocess.call(cmd)
    # cmd = ['gzip', '-d', out_path]
    # print('Decompressing ', file_name)
    # subprocess.call(cmd)

def prepare_data_dir(path = './data'):
  if not os.path.exists(path):
    os.mkdir(path)

if __name__ == '__main__':
  args = parser.parse_args()
  prepare_data_dir()

  if 'celebA' in args.datasets:
    download_celeb_a('./data')
    preprocess_celeb_a('./data')
  if 'lsun' in args.datasets:
    download_lsun('./data')
  if 'mnist' in args.datasets:
    download_mnist('./data')