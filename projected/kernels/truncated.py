from itertools import permutations, product
from enum import Enum, auto

import torch
from torch.nn.parameter import Parameter
from gpytorch.kernels import Kernel
from gpytorch.settings import trace_mode

from typing import List
import numpy as np
from tqdm import tqdm


def precompute_dp_truncated(d: int, max_j: int, max_m: int):
    r"""
    Computes the normalized dynamical programming table, filled with values
    :math:`a_{k, m} / {d \choose k}`, where :math:`m <= d, k <= d`.
    """
    dp = np.zeros((max_j + 1, max_m + 1), dtype=np.float32)
    dp_temp = np.zeros((max_j + 1, max_m + 1), dtype=np.float32)

    # base: d = 1
    dp[0, 0] = 1
    dp[0, 1] = 1
    dp[1, 0] = 1
    dp[1, 1] = -1

    for n in range(2, d+1): #This loop increments over d
        # do some dp temp initialization
        j_iter = min(n, max_j) + 1
        m_iter = min(n, max_m) + 1

        dp_temp[: (j_iter + 1), 0] = 1
        dp_temp[0, : (m_iter + 1)] = 1
        # if n <= max_j:
        #     dp_temp[j_iter, :(m_iter + 1)] = (-1) ** (np.arange(m_iter + 1) % 2)

        arr_j = np.arange(1, j_iter)
        dp_temp[1:j_iter, 1:] = dp[1:j_iter, :-1] - arr_j[:, None] * (dp[1: j_iter, :-1] + dp[: (j_iter-1), :-1]) / n
        dp[:] = dp_temp
    return torch.tensor(dp)


class GraphKernelTruncated(Kernel):

    BASE_PARAMS = {'heat': ['sigma2', 'kappa'], 'matern': ['sigma2', 'kappa', 'nu']}

    r"""
    Computes the Graph Kernel between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`

    :param n: int, size of the graphs (needed for precomputing)
    :param mode: KernelType, the type of the kernel (either HEAT, or MATERN)
    :param sigma2: float, parameter for the kernel
    :param kappa: float, parameter for the kernel
    :param nu: float, parameter for the kernel
    """

    def __init__(self, 
                 n: int, 
                 max_j: int, 
                 mode: str, 
                 max_m: int = None, 
                 is_undirected: bool = False,
                 load_precomputed: bool = True, 
                 trainable_params: List = None, 
                 **kwargs):
        super(GraphKernelTruncated, self).__init__(**kwargs)
        if mode not in ['heat', 'matern']:
            raise RuntimeError("Kernel not supported")

        self.mode = mode
        self.n = n
        if is_undirected:
            self.d = n * (n - 1) // 2
        else:
            self.d = n * n
        
        self.max_j = max_j
        self.max_m = max_m
        if self.max_m is None:
            self.max_m = self.d

        print(f"n={n}, d={self.d}, max_j={self.max_j}, max_m={self.max_m}")
        self.is_undirected = is_undirected

        param_dict = {}
        if trainable_params is None:
            param_dict['sigma2'] = Parameter(torch.tensor(kwargs['sigma2']))
            param_dict['kappa'] = Parameter(torch.tensor(kwargs['kappa']))
            if mode == 'matern':
                param_dict['nu'] = Parameter(torch.tensor(kwargs['nu']))
        else:
            for param_name in trainable_params:
                param_dict[param_name] = Parameter(torch.tensor(kwargs[param_name]))
            
            # Set the non-trainable params as generic attributes of the kernel
            fixed_params = list(set(self.BASE_PARAMS[mode]).difference(set(trainable_params)))
            for param_name in fixed_params:
                setattr(self, param_name, torch.tensor(kwargs[param_name]))
        
        for param_name, value in param_dict.items():
            self.register_parameter(param_name, value)

        self._precompute_dp(load_precomputed)

    def update_phi(self):
        if self.mode == "heat":
            logphi = -self.kappa**2 * torch.arange(self.max_j + 1)
        
        elif self.mode == "matern":
            logphi = -self.nu * torch.log(
                ((2 * self.nu) / (self.kappa**2)) + 2 * torch.arange(self.max_j + 1)
            )

        lse_inputs = torch.zeros(self.max_j + 1, dtype=torch.float32)
        for i in range(0, self.max_j + 1):
            lse_inputs[i] = logphi[i] + self.logfac[self.d] - self.logfac[self.d - i] - self.logfac[i]
        lse_input_max = lse_inputs.max()
        log_c_phi = lse_input_max + torch.log(torch.sum(torch.exp(lse_inputs - lse_input_max)))

        d_arr = torch.full(size=(self.max_j + 1,), fill_value=self.d)
        k_arr = torch.arange(self.max_j + 1)
        d_minus_k_arr = d_arr - k_arr
        logphi_norm = logphi + self.logfac[d_arr] - self.logfac[d_minus_k_arr] - self.logfac[k_arr] - log_c_phi

        self.phi = self.sigma2 * torch.exp(logphi_norm)
        self.dpp = self.dp.T.matmul(self.phi).flatten()

    def _precompute_dp(self, load_precomputed: bool = False):
        if not load_precomputed:
            self.dp = precompute_dp_truncated(d=self.d, max_j=self.max_j, max_m=self.max_m)
        else:
            self.dp = torch.load(f"dps/d={self.d}-max_j={self.max_j}-max_m={self.max_m}.pt")
        
        # We precompute the logfac for large d, since it can be slow doing it inside update_phi()
        logfac = torch.zeros(self.d + 1, dtype=torch.float32)
        for i in range(2, self.d+1):
            logfac[i] = logfac[i - 1] + torch.log(torch.tensor(i).float())
        self.logfac = logfac

    def kern(self, x1: torch.Tensor, x2: torch.Tensor):
        if len(self.batch_shape):
            mul = torch.bitwise_xor(x1[:, :, None].long(), x2[:, None].long())
            mul = mul.sum(dim=[-1])
        else:
            mul = torch.bitwise_xor(x1[:, None].long(), x2[None].long())
            if len(x1.shape) > 2:
                mul = mul.sum(dim=[-2, -1])
            else:
                mul = mul.sum(dim=[-1])

        if self.is_undirected:
            mul = mul / 2

        return self.dpp[mul.long()]


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        """
        Computes the covariance between x1 and x2.
        :param x1: (Tensor `b1 x n x n`), First set of data
        :param x2: (Tensor `b2 x n x n`), Second set of data

        :returns: Tensor of size `b1 x b2`
        """
        # if (
        #     x1.requires_grad
        #     or x2.requires_grad
        #     or trace_mode.on()
        #     or params.get("last_dim_is_batch", False)
        # ):
        #     raise NotImplementedError("Don't know how to handle this case!")
        self.update_phi()
        if len(self.batch_shape):
            kern = self.kern(x1, x2)
            if diag:
                return kern.diagonal(dim1=-2, dim2=-1)
            return kern

        if diag:
            return self.kern(x1, x2).diagonal(dim1=0, dim2=1)
        return self.kern(x1, x2)


class ProjectedKernel(GraphKernelTruncated):
    r"""
    Computes the Graph Quotient Kernel between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`

    :param n: int, size of the graphs (needed for precomputing)
    :param n_approx: int, if 0, uses the full S_n, otherwise, uses this many random permutations to approximate the kernel.
    :param mode: KernelType, the type of the kernel (either HEAT, or MATERN)
    :param equiv: list, a list of lists, where each list includes indices that are treated as equivalent. E.g., [ [0, 1, 2], [3], [4, 5] ]
    :param sigma2: float, parameter for the kernel
    :param kappa: float, parameter for the kernel
    :param nu: float, parameter for the kernel
    """

    def __init__(self, n: int, mode: str, trainable_params: List[str] = None, 
                 equiv_classes: List[List] = None, n_approx: int = 0, **kwargs):
        super(ProjectedKernel, self).__init__(n=n, mode=mode, trainable_params=trainable_params, **kwargs)
        self.n_approx = n_approx
        if equiv_classes is None:
            self.equiv_classes = [list(range(n))]
        else:
            self.equiv_classes = equiv_classes
        self.cache = {}

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        """
        Computes the covariance between x1 and x2.
        :param x1: (Tensor `n x d`), First set of data
        :param x2: (Tensor `m x d`), Second set of data

        :returns: Tensor of size `n x m`
        """
        # if (
        #     x1.requires_grad
        #     or x2.requires_grad
        #     or trace_mode.on()
        #     or params.get("last_dim_is_batch", False)
        # ):
        #     raise NotImplementedError("Don't know how to handle this case!")
        self.update_phi()


        val = torch.zeros((x1.size(0), x2.size(0)), dtype=torch.float32)
        x1 = x1.reshape(x1.size(0), self.n, self.n)
        x2 = x2.reshape(x2.size(0), self.n, self.n)

        if self.n_approx == 0:
            perms = [permutations(e, r=len(e)) for e in self.equiv_classes]

            for i, _perm in enumerate(product(*perms)):
                perm = [v for p in _perm for v in p]
                val = (val * i + self.kern(x1, x2[:, perm, :][:, :, perm])) / (i + 1)

            return val

        else:
            shape_id = (x1.shape, x2.shape)
            if shape_id not in self.cache:
                rand_perms = [
                    [v for e in self.equiv_classes for v in np.array(e)[np.random.permutation(len(e))]]
                    for _ in range(self.n_approx)
                ]

                ms = torch.empty((self.n_approx, self.n_approx, x1.size(0), x2.size(0)), dtype=torch.long)
                
                for i, perm_x1 in tqdm(enumerate(rand_perms)):
                    for j, perm_x2 in enumerate(rand_perms):
                         x1_perm = x1[:, :, perm_x1][:, perm_x1, :]
                         x2_perm = x2[:, :, perm_x2][:, perm_x2, :]
                         ms[i, j, :, :] = x1_perm[:, None].bitwise_xor(x2_perm[None]).sum(dim=(-1, -2)) // 2
                self.cache[shape_id] = ms

            return self.max_jpp[self.cache[shape_id]].sum(dim=[0, 1]) / self.n_approx**2
