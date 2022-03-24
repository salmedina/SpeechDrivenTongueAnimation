# -*- coding: utf-8 -*-
'''
author: Salvador Medina
'''
import argparse
from pathlib import Path

import numpy as np
import torch
from fairseq.models.wav2vec import Wav2VecModel
from tqdm import tqdm

from utils import load_audio, init_torch_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str,
                        help='Directory with the wav files')
    parser.add_argument('-m', '--model_path', type=str,
                        help='Path to the wav2vec model checkpoint file')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Directory for the output npy files with the wav2vec features')
    parser.add_argument('-f', '--feature_type', type=str, default='c',
                        help='Wav2Vec feature selection: {z, c}')
    parser.add_argument('-gid', '--gpu_id', type=int, default=0,
                        help='GPU id under which the inference will be done')
    return parser.parse_args()


def main(opts):
    gpu_id = 0
    input_dir_path = Path(opts.input_dir)
    output_dir_path = Path(opts.output_dir)
    
    # Load wav2vec model
    device = init_torch_device(gpu_id)
    cp = torch.load(opts.model_path, map_location=device)
    model = Wav2VecModel.build_model(cp['args'], task=None).to(device)
    model.load_state_dict(cp['model'])
    model.eval()

    with torch.no_grad():
        pbar = tqdm(list(input_dir_path.glob('*.wav')))
        for wav_path in pbar:
            pbar.set_description(Path(wav_path).name)
            output_path = Path(output_dir_path / wav_path.stem).with_suffix('.npy')
            audio_signal, _ = load_audio(wav_path)
            audio_tensor = torch.Tensor(audio_signal).unsqueeze(0).to(device)
            feat = model.feature_extractor(audio_tensor)
            if opts.feature_type == 'c':
                feat = model.feature_aggregator(feat)
            
            feat_arr = feat.squeeze(0).transpose(0, 1).detach().cpu().numpy()

            # Stack two contiguous features to match 50 FPS,
            # as each feature represents 10 ms
            feat_size = feat_arr.shape[-1]
            # Leave the odd one out :(
            if len(feat_arr) % 2 != 0:
                feat_arr = feat_arr[:-1]
            reshaped_feats = feat_arr.reshape(-1, 2, feat_size)
            stacked_feats = np.column_stack((reshaped_feats[:, 0], reshaped_feats[:, 1]))
           
            np.save(output_path, stacked_feats)


if __name__ == '__main__':
    opts = parse_args()
    main(opts)
