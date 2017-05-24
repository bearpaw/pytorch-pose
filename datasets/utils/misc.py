import torch 
import math
import numpy as np
import matplotlib.pyplot as plt

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
    return torch.from_numpy(h).float()

def draw_gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    # Check that any part of the gaussian is in-bounds
    # pt = pt-1 # 0 index
    tmp_size = math.ceil(3*sigma)
    ul = torch.Tensor([math.floor(pt[0] - tmp_size), math.floor(pt[1] - tmp_size) + 1])
    br = torch.Tensor([math.floor(pt[0] + tmp_size), math.floor(pt[1] + tmp_size) + 1])

    # If not, return the image as is
    if (ul[0] > img.size(1) or ul[1] > img.size(0) or br[0] < 0 or br[1] < 0):
        return img    

    # Generate gaussian
    size = 2 * tmp_size + 1
    g = gaussian((size, size), sigma)

    # Usable gaussian range
    g_x = torch.Tensor([max(0, -ul[0]), min(br[0], img.size(1)) - max(0, ul[0]) + max(0, -ul[0])]).int()
    g_y = torch.Tensor([max(0, -ul[1]), min(br[1], img.size(0)) - max(0, ul[1]) + max(0, -ul[1])]).int()

    # Image range
    img_x = torch.Tensor([max(0, ul[0]), min(br[0], img.size(1))]).int()
    img_y = torch.Tensor([max(0, ul[1]), min(br[1], img.size(0))]).int()

    if g_y[0] == g_y[1] or g_x[0] == g_x[1] or img_y[0] == img_y[1] or img_x[0] == img_x[1]:
            return img
    # assert(g_x[0] >= 0 and g_y[0] >= 0)
    # print("g")
    # print(g_x)
    # print(g_y)
    # print("img")
    # print(img_x)
    # print(img_y)
    # print(g[g_y[0]:g_y[1], g_x[0]:g_x[1]].size())
    # print(img[img_y[0]:img_y[1], img_x[0]:img_x[1]].size())
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]].copy_(g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return img

# def draw_gaussian(img, pt, sigma):
#     # Draw a 2D gaussian
#     # Check that any part of the gaussian is in-bounds
#     # pt = pt-1 # 0 index
#     x, y = pt[0], pt[1]
#     map_h, map_w = img.size(0), img.size(1)
#     if x < 0 or y < 0 or x > map_w or y > map_h:
#         return img

#     # ul = torch.Tensor([math.floor(pt[0] - tmpSize), math.floor(pt[1] - tmpSize)])
#     # br = torch.Tensor([math.floor(pt[0] + tmpSize), math.floor(pt[1] + tmpSize)])

#     # # If not, return the image as is
#     # if (ul[0] > img.size(1) or ul[1] > img.size(0) or br[0] < 0 or br[1] < 0):
#     #     return img    

#     # Generate gaussian
#     tmp_size = math.ceil(3*sigma)
#     hsize = 2 * tmp_size + 1
#     hhsize= math.floor(hsize/2);
#     g = gaussian((hsize, hsize), sigma)

#     # compute kernel start index and end index
#     ksx = 0
#     ksy = 0
#     kex = hsize
#     key = hsize

#     # compute map start index and end index
#     msx = x-hhsize
#     msy = y-hhsize
#     mex = x+hhsize+1
#     mey = y+hhsize+1

#     if y-hhsize < 0:
#         ksy = hhsize-y
#         msy = 0

#     if x-hhsize < 1:
#         ksx = hhsize-x
#         msx = 0

#     if y+hhsize > map_h:
#         key = hsize - ((y+hhsize)-map_h)-1
#         mey = map_h

#     if x+hhsize > map_w:
#         kex = hsize - ((x+hhsize)-map_w)-1
#         mex = map_w

#     print('k %d %d %d %d' % (ksx, ksy, kex, key))
#     print('m %d %d %d %d' % (msx, msy, mex, mey))
#     print('k %d %d ' % (kex - ksx, key - ksy))
#     print('m %d %d ' % (mex - msx, mey - msy))
#     # Usable gaussian range
#     g_x = torch.Tensor([ksx, kex]).int()
#     g_y = torch.Tensor([ksy, key]).int()

#     # Image range
#     img_x = torch.Tensor([msx, mex]).int()
#     img_y = torch.Tensor([msy, mey]).int()

#     print(g_x)
#     print(g_y)
#     print(img_x)
#     print(img_y)

#     assert(g_x[1] - g_x[0] == img_x[1] - img_x[0] and  g_y[1] - g_y[0] == img_y[1] - img_y[0])
#     # img[msy:mey, msx:mex].copy_(g[ksy:key, ksx:kex])

#     print(g[g_y[0]:g_y[1], g_x[0]:g_x[1]].size())
#     print(img[img_y[0]:img_y[1], img_x[0]:img_x[1]].size())

#     print(type(img))
#     print(type(g))
#     img[img_y[0]:img_y[1], img_x[0]:img_x[1]].copy_(g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
#     return img

def imshow(img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.axis('off')

def show_joints(img, pts):
    imshow(img)
    
    for i in range(pts.size(0)):
        if pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], 'yo')
    plt.axis('off')