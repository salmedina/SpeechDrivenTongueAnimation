# -*- coding: utf-8 -*-
'''
author: Salvador Medina
'''
import soundfile as sf

def load_audio(path):
    sound, sample_rate = sf.read(path, dtype='int16')
    # Audio normalization
    audio_signal = audio_signal.astype('float32') / 32767 
    if len(audio_signal.shape) > 1:
        if audio_signal.shape[1] == 1:
            audio_signal = audio_signal.squeeze()
        else:
            audio_signal = audio_signal.mean(axis=1)  # mean value across all channels
    return audio_signal, sample_rate