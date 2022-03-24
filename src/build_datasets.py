import argparse
from pathlib import Path

import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feats_dir', type=str,
                        help='')
    parser.add_argument('-e', '--ema_dir', type=str,
                        help='')
    parser.add_argument('-i', '--index_dir', type=str,
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
    num_frames_list = list()
    for sid in tqdm(idx_list):
        # Load the audio features and the EMA data
        feat_data = np.load(Path(feat_dir, f'{sid}.npy'), allow_pickle=True)
        ema_data = np.load(Path(ema_dir, f'{sid}.npy'), allow_pickle=True)

        # Some frames might have been dropped 
        # since we concatenated the audio features
        min_idx_len = min(len(ema_data), len(feat_data))

        num_frames_list.append(min_idx_len)
        x_list.append(feat_data[:min_idx_len])
        y_pos_list.append(ema_data[:min_idx_len])

    return dict(ids=idx_list,
                frames=num_frames_list,
                X=np.array(x_list),
                Y_pos=np.array(y_pos_list))


def main(args: edict) -> None:
    # --- Training dataset ---
    print('Building training dataset')
    train_idx_path = Path(args.index_dir) / 'train.lst'
    train_save_path = Path(args.output_dir) / 'train_dataset.npy'
    train_save_path.parent.mkdir(exist_ok=True)

    train_idx_list = load_index_list(train_idx_path)
    train_dataset = build_data_dict(idx_list=train_idx_list,
                                    feat_dir=args.feats_dir,
                                    ema_dir=args.ema_dir)

    np.save(train_save_path, train_dataset)
    del train_dataset

    # --- Test dataset ---
    print('Building test dataset')
    test_save_path = Path(args.output_dir) / 'test_dataset.npy'
    test_save_path.parent.mkdir(exist_ok=True)
    test_idx_path = Path(args.index_dir) / 'test.lst'
    test_idx_list = load_index_list(test_idx_path)
    test_dataset = build_data_dict(idx_list=test_idx_list,
                                    feat_dir=args.feats_dir,
                                    ema_dir=args.ema_dir)

    np.save(test_save_path, test_dataset)
    del test_dataset


if __name__ == '__main__':
    args = edict(parse_args())
    main(args)
