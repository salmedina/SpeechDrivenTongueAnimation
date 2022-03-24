import argparse
from pathlib import Path

import numpy as np
from easydict import EasyDict as edict
from ruamel.yaml import YAML
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feats_dir', type=str,
                        help='')
    parser.add_argument('-e', '--ema_dir', type=str,
                        help='')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='')

    return parser.parse_args()


def load_index_list(lst_path: str) -> list:
    return [line.strip() for line in open(lst_path, 'r').readlines()]


def build_data_dict(idx_list: list,
                    feat_dir: str,
                    ema_dir: str) -> dict:

    x_list, y_pos_list = list(), list()
    frames_list = list()
    for sid in tqdm(idx_list):
        # Load the audio features and arrange them accordingly
        feat_data = np.load(Path(feat_dir, f'{sid}.npy'), allow_pickle=True)

        # Load the ema data
        ema_data = np.load(Path(ema_dir, f'{sid}.npy'), allow_pickle=True)

        min_idx_len = min(len(ema_data), len(feat_data))

        frames_list.append(min_idx_len)
        x_list.append(feat_data[:min_idx_len])
        y_pos_list.append(ema_data[:min_idx_len])
        

    return dict(ids=idx_list,
                frames=frames_list,
                X=np.array(x_list),
                Y_pos=np.array(y_pos_list))


def main(args: edict) -> None:

    # Training dataset
    print('Building training dataset')
    train_save_path = Path(args.dataset_dir, args.save_path.train)
    train_save_path.parent.mkdir(exist_ok=True)
    train_idx_list = load_index_list(Path(args.dataset_dir,
                                          args.index.train).
                                     absolute().
                                     as_posix())
    train_dataset = build_data_dict(idx_list=train_idx_list,
                                    feat_dir=args.feats_dir,
                                    ema_dir=args.ema_dir)

    np.save(train_save_path, train_dataset)
    del train_dataset

    # Validation dataset
    print('Building validation dataset')
    valid_save_path = Path(args.dataset_dir, args.save_path.valid)
    valid_save_path.parent.mkdir(exist_ok=True)
    valid_idx_list = load_index_list(Path(args.dataset_dir,
                                          args.index.valid).
                                     absolute().
                                     as_posix())
    valid_dataset = build_data_dict(idx_list=valid_idx_list,
                                    feat_dir=args.feats_dir,
                                    ema_dir=args.ema_dir)

    np.save(valid_save_path, valid_dataset)
    del valid_dataset


if __name__ == '__main__':
    args = edict(parse_args())
    main(args)
