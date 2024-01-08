import torch


class doppler_aug(object):

    def __init__(self, max_rel_v = 3):
        self.max_rel_v = max_rel_v

    def __call__(self, sample):
        X,y = sample
        factor = (2*self.max_rel_v*torch.rand(1) + 343 - self.max_rel_v)/343

        freqs = X.shape[-1]
        if factor == 1: # nan prevention maybe?
            factor += 1e-5
        if factor < 1:
            x = torch.arange(freqs)*factor
            w_lower = 1 - (x - x.int())
            X[:,0::2] = X[:,0::2,x.int()]*w_lower + (1 - w_lower)*X[:,0::2,x.int()+1]
        else:
            x = torch.arange(freqs)*factor
            x = x[x < freqs - 1] # make sure we don't query freq values outside of input vector
            w_lower = 1 - (x - x.int())
            
            X[:,0::2,:x.shape[0]] = X[:,0::2,x.int()]*w_lower + (1 - w_lower)*X[:,0::2,x.int()+1]
            X[:,0::2,x.shape[0]:] = 0

        return X,y

class noise_aug(object):

    def __init__(self, noise_ratio = 0.01):
        self.noise_ratio = noise_ratio

    def __call__(self, sample):
        X,y = sample
        X += self.noise_ratio*X.std(dim=(1,2),keepdim=True)*torch.randn(X.shape)
        return X,y