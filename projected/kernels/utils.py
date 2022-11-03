import torch
from typing import List
import numpy as np
from projected.kernels.base import GraphKernel, ProjectedKernel

KERNELS = {'graph': GraphKernel, 'projected': ProjectedKernel}


def build_kernel(args, n: int, equiv_classes: List[List] = None, 
                 is_undirected: bool = False, classification: bool = False):
    kernel_cls = KERNELS.get(args.kernel, None)
    if kernel_cls is None:
        raise ValueError()
    
    kernel_params = {"kappa": args.kappa}
    if args.kernel_mode == "matern":
        if is_undirected:
            add_to_nu = n * (n - 1) / 4
        else:
            add_to_nu = n * n / 2
        kernel_params["nu"] = args.nu + add_to_nu

    if args.kernel == "projected":
        kernel_params['equiv_classes'] = equiv_classes
        kernel_params["n_approx"] = args.n_approx

    if classification:
        if args.strategy == "lmc":
            kernel_params['batch_shape'] = torch.Size([args.n_latent])

    kernel = kernel_cls(n=n, mode=args.kernel_mode, is_undirected=is_undirected, 
                        trainable_params=args.trainable_params, **kernel_params)
    return kernel


def apply_cantor_fn(n):

    def cantor(n):
        return [0.] + cant(0., 1., n) + [1.]

    def cant(x, y, n):
        if n == 0:
            return []

        new_pts = [2.*x/3. + y/3., x/3. + 2.*y/3.]
        return cant(x, new_pts[0], n-1) + new_pts + cant(new_pts[1], y, n-1)

    return cantor(n)
