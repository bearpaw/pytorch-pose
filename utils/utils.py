import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import scipy.ndimage as ndi
import scipy.misc as misc
import math
import numpy as np
from numpy import linalg as LA
import pdb
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import sys



def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray
    
def gaussian(shape=(7,7),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return torch.from_numpy(h)

def load_image(img_path):
    # return torch.from_numpy(np.array(Image.open(img_path).convert('RGB')).transpose([2, 0, 1])).float()
    im = np.transpose(misc.imread(img_path), (2, 0, 1)) # H x W x C => C x H x W
    im = torch.from_numpy(im).float()/255               # normalize to [0, 1]
    return im

def imshow(img):
    # if img.max() < 2:
    #     img.mul_(255)
    # npimg = img.int().numpy()*255
    # print(npimg)
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

def resize(img, owidth, oheight):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale((owidth, oheight)),
        transforms.ToTensor(),
    ])(img)

def upsample(x, scale_factor=1):   
    assert x.dim() == 4, 'Input should be a 4-dim tensor'
    model = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    output = model(Variable(x))
    output = output.data
    return output

def downsample(x, scale_factor=1):    
    model = nn.AvgPool2d(scale_factor, stride=scale_factor)
    output = model(Variable(x))
    return output.data

def rotate(img, angle):
    return torch.from_numpy(ndi.rotate(img.numpy(), angle, axes=(2, 1), reshape=False))


def get_transform(center, scale, rot, res):
    h = 200 * scale
    t = torch.eye(3)

    # Scaling
    t[0][0] = res*1.0 / h
    t[1][1] = res*1.0 / h

    # Translation
    t[0][2] = res*1.0 * (-center[0] / h + .5)
    t[1][2] = res*1.0 * (-center[1] / h + .5)


    # Rotation
    if rot != 0:
        rot = -rot
        r = torch.eye(3)
        ang = rot * math.pi / 180
        s = math.sin(ang)
        c = math.cos(ang)
        r[0][0] = c
        r[0][1] = -s
        r[1][0] = s
        r[1][1] = c
        # Need to make sure rotation is around center
        t_ = torch.eye(3)
        t_[0][2] = -res/2
        t_[1][2] = -res/2
        t_inv = torch.eye(3)
        t_inv[0][2] = res/2
        t_inv[1][2] = res/2
        t = t_inv.mm(r.mm(t_.mm(t)))

    return t

def transform(pt, center, scale, rot, res, invert=False):
    pt_ = torch.ones(3, 1)
    pt_[0],pt_[1] = pt[0]-1,pt[1]-1

    t = get_transform(center, scale, rot, res)
    if invert:
        t = torch.inverse(t)

    new_point = t.mm(pt_)[0:2]

    return new_point.int().squeeze()+1


def crop(img, center, scale, rot, res):
    ndim = img.dim()
    if ndim == 2:
        img.unsqueeze_(0)
    ht,wd = img.size(1), img.size(2)
    tmpImg, newImg = img, np.zeros((img.size(0), res, res))


    # Modify crop approach depending on whether we zoom in/out
    # This is for efficiency in extreme scaling cases
    scaleFactor = (200.0 * scale) / res

    if scaleFactor < 2: 
        scaleFactor = 1
    else:
        newH = int(math.floor(ht/ scaleFactor))
        newW = int(math.floor(wd/ scaleFactor))

        if max(newH, newW) < 2:
           # Zoomed out so much that the image is now a single pixel or less
           if ndim == 2:
                newImg = newImg.squeeze()
           return newImg
        else:
           tmpImg = resize(img, newW, newH)
           ht,wd = tmpImg.size(1), tmpImg.size(2)
    # Calculate upper left and bottom right coordinates defining crop region
    c,s = center/scaleFactor, scale/scaleFactor
    ul = transform([1,1], c, s, 0, res, invert=True)
    br = transform([res+1,res+1], c, s, 0, res, invert=True)

    if scaleFactor >= 2: 
        br = ul + res
    # If the image is to be rotated, pad the cropped area
    pad = int(math.ceil((torch.norm((ul-br).float())-(br[0]-ul[0]))/2))

    if rot != 0:
        ul.add_(-pad)
        br.add_(pad)

    # Define the range of pixels to take from the old image
    old_ = torch.IntTensor([max(1, ul[1]), min(br[1], ht+1) - 1,
                       max(1, ul[0]), min(br[0], wd+1) - 1])
    # And where to put them in the new image
    new_ = torch.IntTensor([max(1, -ul[1] + 2), min(br[1], ht+1) - ul[1],
                       max(1, -ul[0] + 2), min(br[0], wd+1) - ul[0]])

    # Initialize new image and copy pixels over
    newImg = torch.zeros(img.size(0), br[1] - ul[1], br[0] - ul[0])

    try:
        newImg[:, new_[0]-1:new_[1], new_[2]-1:new_[3]].copy_(tmpImg[:, old_[0]-1:old_[1], old_[2]-1:old_[3]])
    except:
        print('Error occurred during cropping')
        raise

    # Rotate the image and remove padded area
    if rot != 0:
        print(newImg.size())
        newImg = rotate(newImg, rot)
        print(newImg.size())
        newImg = newImg[:, pad:newImg.size(1)-pad, pad:newImg.size(2)-pad].clone()


    if scaleFactor < 2:
        newImg = resize(newImg,res,res)
    if ndim == 2: 
        newImg = newImg.squeeze()

    return newImg

def drawGaussian(img, pt, sigma):
    # Draw a 2D gaussian
    # Check that any part of the gaussian is in-bounds
    # pt = pt-1 # 0 index
    tmpSize = math.ceil(3*sigma)
    ul = torch.Tensor([math.floor(pt[0] - tmpSize), math.floor(pt[1] - tmpSize)])
    br = torch.Tensor([math.floor(pt[0] + tmpSize), math.floor(pt[1] + tmpSize)])

    # If not, return the image as is
    if (ul[0] > img.size(1) or ul[1] > img.size(0) or br[0] < 0 or br[1] < 0):
        return img    

    # Generate gaussian
    size = 2*tmpSize + 1
    g = gaussian((size, size), sigma)


    # Usable gaussian range
    g_x = torch.Tensor([max(0, -ul[0]), min(br[0], img.size(1)) - max(0, ul[0]) + max(0, -ul[0])]).int()
    g_y = torch.Tensor([max(0, -ul[1]), min(br[1], img.size(0)) - max(0, ul[1]) + max(0, -ul[1])]).int()
    # Image range
    img_x = torch.Tensor([max(0, ul[0]), min(br[0], img.size(1))]).int()
    img_y = torch.Tensor([max(0, ul[1]), min(br[1], img.size(0))]).int()

    assert(g_x[1] > 0 and g_y[1] > 0)
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]].copy_(g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return img



_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
