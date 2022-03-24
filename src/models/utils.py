# -*- coding: utf-8 -*-
"""
@author: Salvador Medina
"""

import torch

__all__ = ['save_checkpoint', 'load_checkpoint', 'print_checkpoint',
           'save_model', 'load_model',
           'count_total_params', 'count_trainable_params']

def save_checkpoint(epoch, model, model_params, optimizer, optimizer_params,
                    loss, global_step, save_path):
    torch.save({'epoch': epoch,
                'model_class': model.__class__.__name__,
                'model_params': model_params,
                'model_state_dict': model.state_dict(),
                'optimizer_class': optimizer.__class__.__name__,
                'optimizer_params': optimizer_params,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'global_step': global_step
                }, save_path)


def load_checkpoint(checkpoint_path, model_class, optimizer_class, device):
    checkpoint = torch.load(checkpoint_path, map_location=str(device))

    # Load model
    model = model_class(**checkpoint['model_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Load optimizer
    optimizer = optimizer_class(model.parameters(), **checkpoint['optimizer_params'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load log status
    log_step = checkpoint['global_step']

    return model, optimizer, log_step


def save_model(epoch, model, model_params, save_path):
    torch.save({'epoch': epoch,
                'model_class': model.__class__.__name__,
                'model_params': model_params,
                'model_state_dict': model.state_dict()
                }, save_path)


def load_model(checkpoint_path, model_class, device):
    checkpoint = torch.load(checkpoint_path, map_location=str(device))

    # TODO: Hack to remove drop path rate in tongueformer
    if 'drop_path_rate' in checkpoint['model_params']:
        checkpoint['model_params']['drop_path_rate'] = 0.0

    # Load model
    model = model_class(**checkpoint['model_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model


def print_checkpoint(checkpoint_path, name, params_dict):
    checkpoint = torch.load(checkpoint_path)

    print(f'{"Name":>13}: {checkpoint["model_name"]}')
    for k, v in checkpoint['model_params'].items():
        print(f'{k:>13}: {v}')


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)