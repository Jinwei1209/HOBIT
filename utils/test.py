import numpy as np
import math
import torch
import torch.nn.functional as F
import scipy.ndimage

from torch.autograd import Variable
from math import exp
from utils.data import *
from utils.medi import *

def compute_rmse(chi_recon, chi_true, mask_csf):
    """chi_true is the refernce ground truth"""
    mask = abs(chi_true) > 0
    chi_recon = (chi_recon - np.mean(chi_recon[mask_csf==1])) * mask
    chi_true = (chi_true - np.mean(chi_true[mask_csf==1])) * mask
    return np.sqrt(np.sum((chi_recon - chi_true)**2)/np.prod(chi_recon.shape))

def compute_fidelity_error(chi, rdf, voxel_size):
    """data consistenty loss, with rdf the measured data"""
    D = dipole_kernel(rdf.shape, voxel_size, [0, 0, 1])
    mask = abs(rdf) > 0
    diff = np.fft.ifftn( np.fft.fftn(chi) * D ) - rdf
    return np.sqrt(np.sum((diff*mask)**2))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

# 3D SSIM
def compute_ssim(img1, img2, mask_csf, window_size = 11, size_average = True):

    img1.to('cuda:0')
    img2.to('cuda:0')
    
    mask = abs(img2) > 0
    img1 = (img1 - torch.mean(img1[mask_csf==1])) * mask
    img2 = (img2 - torch.mean(img2[mask_csf==1])) * mask

    #  modified to make sure dynamic range is 0-255 for QSM
    min_img = torch.min(torch.min(img1, torch.min(img2)))
 
    img1[img1!=0] = img1[img1!=0] - min_img
    img2[img2!=0] = img2[img2!=0] - min_img
    
    max_img = torch.max(torch.max(img1, torch.max(img2)))
    
    img1 = 255 * img1 / max_img
    img2 = 255 * img2 / max_img

    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def compute_hfen(img1, img2, mask):
     
    # Laplacian of Gaussian filter to get high frequency information:
    filt_siz = np.array([1, 1, 1]) * 15
    sig = np.array([1, 1, 1]) * 1.5
    
    siz = (filt_siz - 1) / 2
    x, y, z = np.meshgrid(np.arange(-siz[0], siz[0], 1), np.arange(-siz[1], siz[1], 1), np.arange(-siz[2], siz[2], 1))

    h = np.exp(-(x*x/2/sig[0]**2 + y*y/2/sig[1]**2 + z*z/2/sig[2]**2))
    h = h / np.sum(h)
    
    arg = (x*x/sig[0]**4 + y*y/sig[1]**4 + z*z/sig[2]**4 - (1/sig[0]**2 + 1/sig[1]**2 + 1/sig[2]**2))
    H = arg * h
    H = H - np.sum(H) / np.prod(2*siz+1)

    img1_log = scipy.ndimage.convolve(img1, H) * mask
    img2_log = scipy.ndimage.convolve(img2, H) * mask
    return 100 * np.sqrt(np.sum((img1_log - img2_log)**2)) / np.sqrt(np.sum(img2_log**2))

def compute_R_ICH(img1, img2, mask):

    # Shadow artifact quantification surrounding ICH
    return (np.std(img1[mask==1]) - np.std(img2[mask==1])) / np.std(img2[mask==1])


