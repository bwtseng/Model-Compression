import pandas as pd 
from tabulate import tabulate 
import torch
import distiller
import logging
msglogger = logging.getLogger()

def weights_sparsity_summary(model, return_total_sparsity=False, param_dims=[2, 4]):
    df = pd.DataFrame(columns=['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)',
                               'Cols (%)', 'Rows (%)', 'Ch (%)', '2D (%)', '3D (%)',
                               'Fine (%)', 'Std', 'Mean', 'Abs-Mean'])
    pd.set_option('precision', 2)
    params_size = 0
    sparse_params_size = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            _density = distiller.density(param) ## From: non zero value/tensor size.
            params_size += torch.numel(param) # tensor size
            sparse_params_size += param.numel() * _density # number of non zero value
            df.loc[len(df.index)] = ([
                name,
                distiller.size_to_str(param.size()),
                torch.numel(param),
                int(_density * param.numel()),
                distiller.sparsity_cols(param)*100,
                distiller.sparsity_rows(param)*100,
                distiller.sparsity_ch(param)*100,
                distiller.sparsity_2D(param)*100,
                distiller.sparsity_3D(param)*100,
                (1-_density)*100,
                param.std().item(),
                param.mean().item(),
                param.abs().mean().item()
            ])

    total_sparsity = (1 - sparse_params_size/params_size)*100

    df.loc[len(df.index)] = ([
        'Total sparsity:',
        '-',
        params_size,
        int(sparse_params_size),
        0, 0, 0, 0, 0,
        total_sparsity,
        0, 0, 0])

    if return_total_sparsity:
        return df, total_sparsity
    return df


def weights_sparsity_tbl_summary(model, return_total_sparsity=False, param_dims=[2, 4]):
    df, total_sparsity = weights_sparsity_summary(model, return_total_sparsity=True, param_dims=param_dims)
    df.to_csv("summary.csv", index=False)
    t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
    if return_total_sparsity:
        return t, total_sparsity
    return t


def masks_sparsity_summary(model, scheduler, param_dims=[2, 4]):
    df = pd.DataFrame(columns=['Name', 'Fine (%)'])
    pd.set_option('precision', 2)
    params_size = 0
    sparse_params_size = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            mask = scheduler.zeros_mask_dict[name].mask
            if mask is None:
                _density = 1
            else:
                _density = distiller.density(mask)
            params_size += torch.numel(param)
            sparse_params_size += param.numel() * _density
            df.loc[len(df.index)] = ([name, (1-_density)*100])

    assert params_size != 0
    total_sparsity = (1 - sparse_params_size/params_size)*100
    df.loc[len(df.index)] = (['Total sparsity:', total_sparsity])
    return df


def masks_sparsity_tbl_summary(model, scheduler, param_dims=[2, 4]):
    df = masks_sparsity_summary(model, scheduler, param_dims=param_dims)
    return tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")