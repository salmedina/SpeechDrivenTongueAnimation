# -*- coding: utf-8 -*-
'''
author: Salvador Medina
'''
import os
import random
import numpy as np
import torch
import soundfile as sf


def seed_everything(seed: int) -> None:
    """ Seed all random number generators

    Args:
        seed (int): Seed number for all random numbergenerators
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_torch_device(gpu_id: int=-1) -> torch.device:
    """ Creates torch device based on the GPU ID

    Args:
        gpu_id (int, optional): CUDA device index, CPU if < 0
    """
    device_str = 'cpu'
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if gpu_id >= 0 and gpu_id < num_gpus:
            device_str = f'cuda:{gpu_id}'

    return torch.device(device_str)


def load_audio(path: str) -> tuple:
    """ Load and normalize audio file

    Args:
        path (str): Path to the audio file

    Returns:
        np.array, int: audio signal array, sample rate
    """
    audio_signal, sample_rate = sf.read(path, dtype='int16')
    # Audio normalization
    audio_signal = audio_signal.astype('float32') / 32767 
    if len(audio_signal.shape) > 1:
        if audio_signal.shape[1] == 1:
            audio_signal = audio_signal.squeeze()
        else:
            audio_signal = audio_signal.mean(axis=1)  # mean value across all channels
    return audio_signal, sample_rate


def calc_seq_dist(gt_tensor: torch.Tensor,
                  pred_tensor: torch.Tensor) -> float:
    """ Calculates the sequence MSE from tensors

    Args:
        gt_tensor (torch.Tensor): ground truth
        pred_tensor (torch.Tensor): predictions values

    Returns:
        float: MSE
    """
    # Reshape the tensors to 3D coords
    gt_tensor = gt_tensor.reshape(-1, 10, 3)
    pred_tensor = pred_tensor.reshape(-1, 10, 3)

    # Calculate batch L2 error
    # -- Axis; 0: frames, 1:joints, 2:coords
    pose_delta = gt_tensor - pred_tensor
    error_tensor = torch.sqrt(torch.sum(
                              torch.mul(pose_delta, pose_delta),
                              axis=2))
    # mean across the joints through the sequence,
    # then the mean across the joints
    joints_error = error_tensor.mean(axis=0)
    sample_error = joints_error.mean().cpu().item()

    return sample_error