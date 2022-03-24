# -*- coding: utf-8 -*-
'''
author: Salvador Medina
'''
import os
import random
import numpy as np
import torch
import soundfile as sf


def seed_everything(seed: int):
    """ Seed all random number generators

    Args:
        seed (int): Seed number for all random numbergenerators
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_torch_device(gpu_id=-1):
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


def load_audio(path):
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