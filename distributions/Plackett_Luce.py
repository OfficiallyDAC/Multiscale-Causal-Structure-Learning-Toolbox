#adapted from PyTorch implementation --> https://github.com/agadetsky/pytorch-pl-variance-reduction

import torch
import torch.nn.functional as F

from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution

def logcumsumexp(x, dim):
    # slow implementation, but ok for now
    if (dim != -1) or (dim != x.ndimension() - 1):
        x = x.transpose(dim, -1)

    out = []
    for i in range(1, x.size(-1) + 1):
        out.append(torch.logsumexp(x[..., :i], dim=-1, keepdim=True))
    out = torch.cat(out, dim=-1)

    if (dim != -1) or (dim != x.ndimension() - 1):
        out = out.transpose(-1, dim)
    return out


def reverse_logcumsumexp(x, dim):
    return torch.flip(logcumsumexp(torch.flip(x, dims=(dim, )), dim), dims=(dim, ))

def smart_perm(x, permutation):
    assert x.size() == permutation.size()

    if x.ndimension() == 1:
        ret = x[permutation]
    elif x.ndimension() == 2:
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            permutation.flatten()
        ].view(d1, d2)
    elif x.ndimension() == 3:
        d1, d2, d3 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2 * d3)).flatten(),
            torch.arange(d2).unsqueeze(1).repeat((1, d3)).flatten().unsqueeze(0).repeat((1, d1)).flatten(),
            permutation.flatten()
        ].view(d1, d2, d3)
    elif x.ndimension() == 4:
        d1, d2, d3, d4 = x.size()
        ret = x[torch.arange(d1).unsqueeze(1).repeat((1,d2*d3*d4)).flatten(),
            torch.arange(d2).unsqueeze(1).repeat((1, d3*d4)).flatten().unsqueeze(0).repeat((1, d1)).flatten(),
            torch.arange(d3).unsqueeze(1).repeat((1,d4)).flatten().unsqueeze(0).repeat((1,d1*d2)).flatten(),
            permutation.flatten()
            ].view(d1,d2,d3,d4) 
    else:
        ValueError("Only 4 dimensions maximum")
    return ret

def make_permutation_matrix(b):
    # permutation matrix P_b with column representation: p_{ij} = 1 if j = b(i)
    return torch.eye(b.size(-1))[b].squeeze(dim=0)

def _to_z_tilde(logits, b, v=None):
    '''Produce posterior z
    Parameters
    ----------
    logits : torch.Tensor
    b : torch.Tensor
    v : torch.Tensor (you can specify noise yourself)
    Returns
    -------
    z : torch.Tensor
    '''
    if v is not None:
        assert v.size() == logits.size()
    else:
        v = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    b_inv = torch.sort(b, dim=-1)[1]
    log_probs = smart_perm(F.log_softmax(logits, dim=-1), b)
    gumbel = torch.log(-torch.log(v))
    z_tilde = -logcumsumexp(gumbel - reverse_logcumsumexp(log_probs, dim=-1), dim=-1)
    z_tilde = smart_perm(z_tilde, b_inv)
    return z_tilde

class PlackettLuce(TorchDistribution):
    """
        Plackett-Luce distribution
    """
    
    arg_constraints = {"logits": constraints.real_vector}
    has_rsample=False
    
    def __init__(self, logits, validate_args=None):
        # last dimension is for scores of Plackett-Luce
        self.logits = logits
        self.size = self.logits.size()
        #Indices over .batch_shape denote conditionally independent random variables 
        batch_shape = self.logits.shape[:-1]
        #Indices over .event_shape denote dependent random variables (i.e. one draw from a distribution)
        event_shape = self.logits.shape[-1:]
        
        super(PlackettLuce, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PlackettLuce, _instance)
        batch_shape = torch.Size(batch_shape)
        logits_shape = batch_shape + self.event_shape
        new.logits = self.logits.expand(logits_shape)
        new.size = new.logits.size()
        
        super(PlackettLuce, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new
        
    
    def sample(self, sample_shape=torch.Size()):
        # sample permutations using Gumbel-max trick to avoid cycles
        with torch.no_grad():
            if sample_shape:
                logits = self.logits.unsqueeze(0).expand(sample_shape, *self.size)
            else:
                logits = self.logits.unsqueeze(0).expand(1, *self.size)
            u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
            z = self.logits - torch.log(-torch.log(u))
            samples = torch.sort(z, descending=True, dim=-1)[1]
        return samples.squeeze(dim=0)

    def log_prob(self, samples):
        # samples.shape = sample_shape + batch_shape + event_shape
        # samples are permutations not permutation matrices
        if samples.ndimension() == self.logits.ndimension():  # then we already expanded logits
            logits = smart_perm(self.logits, samples)
        elif samples.ndimension() > self.logits.ndimension():  # then we need to expand it here
            logits = self.logits.unsqueeze(0).expand(*samples.size())
            logits = smart_perm(logits, samples)
        else:
            raise ValueError("Something wrong with dimensions")
        logp = (logits - reverse_logcumsumexp(logits, dim=-1)).sum(-1)
        return logp