import numpy as np
import scipy.ndimage.filters as filters
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def init_basis_hermite(sigma,bases,extent):
    filterExtent = extent
    x = np.arange(-filterExtent, filterExtent+1, dtype=np.float)
    imSize = filterExtent*2+1
    impulse = np.zeros( (np.int(imSize), np.int(imSize)) )
    impulse[(np.int(imSize))/2,(np.int(imSize))/2] = 1.0
    nrBasis = 15
    hermiteBasis = np.empty( (np.int(nrBasis), np.int(imSize), np.int(imSize)) )
    g = 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(np.square(x)/(-2*np.square(sigma)))
    g = g/g.sum()
    g1 = sigma * -(x/ np.square(sigma)) * g
    g2 = np.square(sigma) * ( (np.square(x)-np.power(sigma,2)) / np.power(sigma,4)) * g
    g3 = np.power(sigma,3) * -( (np.power(x,3) - 3 * x * np.square(sigma)) / np.power(sigma,6)) * g
    g4 = np.power(sigma,4) * ( ( (np.power(x,4) - 6 *  np.square(x) * np.square(sigma) + 3 * np.power(sigma,4)) / np.power(sigma,8) ) ) * g
    gauss0x = filters.convolve1d(impulse, g, axis=1)
    gauss0y = filters.convolve1d(impulse, g, axis=0)
    gauss1x = filters.convolve1d(impulse, g1, axis=1)
    gauss1y = filters.convolve1d(impulse, g1, axis=0)
    gauss2x = filters.convolve1d(impulse, g2, axis=1)
    gauss0 = filters.convolve1d(gauss0x, g, axis=0)
    hermiteBasis[0,:,:] = gauss0
    vmax = gauss0.max()
    vmin = -vmax
    #print vmax, vmin
    hermiteBasis[1,:,:] = filters.convolve1d(gauss0y, g1, axis=1) # g_x
    hermiteBasis[2,:,:] = filters.convolve1d(gauss0x, g1, axis=0) # g_y
    hermiteBasis[3,:,:] = filters.convolve1d(gauss0y, g2, axis=1) # g_xx
    hermiteBasis[4,:,:] = filters.convolve1d(gauss0x, g2, axis=0) # g_yy
    hermiteBasis[5,:,:] = filters.convolve1d(gauss1x, g1, axis=0) # g_yy
    hermiteBasis[6,:,:] = filters.convolve1d(gauss0y, g3, axis=1) # g_xxx
    hermiteBasis[7,:,:] = filters.convolve1d(gauss0x, g3, axis=0) # g_yyy
    hermiteBasis[8,:,:] = filters.convolve1d(gauss1y, g2, axis=1) # g_xxy
    hermiteBasis[9,:,:] = filters.convolve1d(gauss1x, g2, axis=0) # g_yyx
    hermiteBasis[10,:,:] = filters.convolve1d(gauss0y, g4, axis=1) # g_xxxx
    hermiteBasis[11,:,:] = filters.convolve1d(gauss0x, g4, axis=0) # g_yyyy
    hermiteBasis[12,:,:] = filters.convolve1d(gauss1y, g3, axis=1) # g_xxxy
    hermiteBasis[13,:,:] = filters.convolve1d(gauss1x, g3, axis=0) # g_yyyx
    hermiteBasis[14,:,:] = filters.convolve1d(gauss2x, g2, axis=0) # g_yyxx
    
    return torch.from_numpy(np.asarray(hermiteBasis[0:bases,:,:], dtype=np.float32))

def init_bias(units):
    return torch.from_numpy(np.asarray(np.zeros(units), dtype=np.float32))

def init_alphas(nrFilters,channels,nrBasis):
    return torch.from_numpy(np.asarray(np.random.uniform(low=-1.0,high=1.0,size=(nrFilters,channels,nrBasis)), dtype=np.float32))

def init_weights(shape):
    return torch.from_numpy(np.asarray(np.random.randn(*shape) * 0.01,dtype=np.float32))

def ternarize(tensor):
    output = torch.zeros(tensor.size())
    delta = Delta(tensor)
    alpha = Alpha(tensor,delta)
    for i in range(tensor.size()[0]):
        for w in tensor[i].view(1,-1):
            pos_one = (w > delta[i]).type(torch.FloatTensor)
            neg_one = -1 * (w < -delta[i]).type(torch.FloatTensor)
        out = torch.add(pos_one,neg_one).view(tensor.size()[1:])
        output[i] = torch.add(output[i],torch.mul(out,alpha[i]))
    return output
