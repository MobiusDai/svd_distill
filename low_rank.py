import torch
import numpy as np
import torch.nn as nn

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=32, bias=True):
        super(LowRankLinear, self).__init__()
        self.sv = nn.Parameter(torch.Tensor(rank))
        self.weight_l = nn.Linear(in_features, rank, bias=False)
        self.weight_r = nn.Linear(rank, out_features, bias=bias)

    def init_parameters(self, sv, weight_l, weight_r, bias):
        self.sv.data = sv
        self.weight_l.weight.data = weight_l
        self.weight_r.weight.data = weight_r
        if not bias is None: 
            self.weight_r.bias.data.copy_(bias)

    def forward(self, x):
        x = self.weight_l(x)
        x = x * self.sv
        x = self.weight_r(x)
        return x


def low_rank_approximate(mat_org: torch.tensor, rank=32):
    """ Learning a low-rank decomposition for the given matrix.

    Args:
        mat_org (torch.tensor): the given matrix.
        rank (int, optional): defined rank value. Defaults to 16.
    """
    device = mat_org.device

    if not device == 'cpu':
        mat_org = mat_org.cpu()
    u, s, vh = np.linalg.svd(mat_org.detach().numpy(), full_matrices=True)

    s_val = torch.tensor(s[:rank])
    mat_q = torch.tensor(u[:, :rank])
    mat_r = torch.tensor(vh[:rank, :])
    error = nn.functional.mse_loss(mat_q @ mat_r, mat_org)

    mat_q = mat_q.to(device)
    mat_r = mat_r.to(device)

    output = {'mat_q': mat_q,
              'mat_r': mat_r.t(),
              'sv': s_val,
              'error': error}
    return output


class ModuleLowRank(object):
    """ Replace the original Linear matrix with two low-rank linear matrices.

    Args:
        compress_ratio (int): the pre-defined rank ratio value.
        name_omit (list of str): the omitted name list for low-rank approximation.
        is_approximate (bool, optional): perform low-rank approximation. Defaults to True.
    """

    def __init__(self,
                 compress_ratio=3,
                 name_omit=list(),
                 name_include=list(),
                 is_approximate=True):
        super().__init__()
        # name_omit or name_include
        self.compress_ratio = compress_ratio
        self.name_omit = name_omit
        self.name_include = name_include
        self.is_approximate = is_approximate

    def _apply(self, name: str, module: nn.Linear):
        """ Apply nn.Sequential for replacement of the Linear module.

        Args:
            name (str): module name
            module (nn.Linear): the given Linear module
        """
        shape = (module.in_features, module.out_features)
        weight, bias = module.weight, module.bias

        rank = (shape[0] * shape[1]) // (self.compress_ratio * (shape[0] + shape[1]))
        rank = int(rank)

        lr_out = low_rank_approximate(weight.t(), rank)
        sv, weight_l, weight_r = lr_out['sv'], lr_out['mat_q'], lr_out['mat_r']

        low_rank_module = LowRankLinear(shape[0], shape[1], rank=rank, bias=not bias is None)
        low_rank_module.init_parameters(sv, weight_l, weight_r, bias)

        return {'module_rep': low_rank_module,}

    def __call__(self, module: nn.Module):
        copied_modules = {name: module_sub for name, module_sub in module.named_modules()}
        for name, module_sub in copied_modules.items():
            if isinstance(module_sub, nn.Linear): 
                if self.name_include and not any(n in name for n in self.name_include):
                    continue
                if self.name_omit and any(n in name for n in self.name_omit):
                    continue
                if module_sub.out_features < 10:
                    continue # for some head matrix, such as image-text match head

                base, localname = module, name
                while '.' in localname:
                    prefix, localname = localname.split('.', 1)
                    base = base.__getattr__(prefix)
                output = self._apply(name, module_sub)
                print("applying low rank on", name)

                setattr(base, localname, output['module_rep'])

        return module
