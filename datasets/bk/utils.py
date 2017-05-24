import  torch
import  torch.nn as nn
import  torchvision.transforms as transforms
from    torch.autograd import Variable
import  scipy.ndimage as ndi
import  scipy.misc as misc
import  math
import  numpy as np
import  pdb
import  matplotlib.pyplot as plt





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




# a = torch.zeros(64, 64)
# a = drawGaussian(a, torch.Tensor([10, 10]), 2)

# plt.imshow(a.numpy())
# plt.show()
# a.expand(3, 64, 64)

# imshow(a)
