import argparse
import os
import os.path as osp
import time
from io import open
from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from ruamel.yaml import YAML
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TongueMocapDataset
from logger.model_logger import ModelLogger
from models import GRU, save_checkpoint
from utils import calc_seq_dist, init_torch_device, seed_everything

seed_everything(78373)

loss_iter_str = 'loss/iter'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--local_config_path', type=str,
                        help='Path to the local machine configuration')
    parser.add_argument('-c', '--config_path', type=str,
                        help='Path to the experiment configuration')
    parser.add_argument('-gid', '--gpu_id', type=int, default=0,
                        help='GPU to be used for training')
    return parser.parse_args()


def print_training_header(model, config):
    print('\nTraining Model')
    print('=' * 70)
    print(f'LR:         {config.training.learning_rate}')
    print(f'dropout:    {config.training.dropout}')
    print(f'batch size: {config.training.batch_sz}')
    print(f'model dir:  {config.paths.save_dir}')
    print(f'log dir:    {config.paths.log_dir}')
    print(f'model:{model}')
    print('=' * 70)
    print()


def get_criterion(loss):
    """Creates a criterion from a label for MSE, Huber, and L1, otherwise None

    Args:
        loss_label (str): loss label string

    Returns:
        nn.criterion: pytorch loss calculator
    """
    if loss.label == 'mse':
        return nn.MSELoss()
    elif loss.label == 'huber':
        return nn.SmoothL1Loss()
    elif loss.label == 'l1':
        return nn.L1Loss()

    return None


def train_epoch(epoch, model, optimizer, criterion, data_loader,
                config, model_logger, global_step, device, print_every):
    print('Training phase')
    model.train()

    epoch_start_time = time.time()
    iter_start_time = time.time()
    dataloader_iter = iter(data_loader)
    running_loss = 0.
    batch_idx = 0
    total_loss = 0.0
    iter_times_list = list()
    h0 = model.init_hidden(config.training.batch_sz).to(device)

    for input_tensor, output_tensor in dataloader_iter:
        # skip the batch that is not complete
        if input_tensor.shape[0] != config.training.batch_sz:
            continue
        global_step += 1
        batch_idx += 1

        h = h0.data
        y_hat, _ = model(x=input_tensor.to(device).float(), h=h)
        loss = criterion(y_hat, output_tensor.to(device).float())
        optimizer.zero_grad()
        if batch_idx % print_every == 0:
            model_logger.train.add_scalar(loss_iter_str, loss, global_step)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % print_every == 0:
            iter_time = time.time() - iter_start_time
            iter_times_list.append(iter_time)
            iter_start_time = time.time()
            print(f'[Training] Epoch {epoch}', end='')
            print(f'  Iter Time:{iter_time:.3f}', end='')
            print(f'  Step:{batch_idx}/{len(data_loader):<8}', end='')
            print(f'  l: {running_loss/batch_idx:<10}')

    total_loss = running_loss/len(data_loader)
    epoch_time_str = time.strftime("%H:%M:%S",
                                   time.gmtime(time.time() - epoch_start_time))
    print(f'Totals: Training  loss: {total_loss}  time: {epoch_time_str}')
    print()

    model_save_path = osp.join(config.paths.save_dir,
                               f'{epoch:02d}_{total_loss:06f}.pt')

    save_checkpoint(epoch=epoch,
                    model=model,
                    model_params=model.params,
                    optimizer=optimizer,
                    optimizer_params=dict(lr=config.training.learning_rate),
                    loss=total_loss,
                    global_step=global_step,
                    save_path=model_save_path)

    return global_step


def validate_epoch(epoch, model, optimizer, criterion, data_loader, dataset,
                   config, model_logger, global_step, best, device,
                   print_every):
    with torch.no_grad():
        print('Validation phase')
        model.eval()

        epoch_start_time = time.time()
        iter_start_time = time.time()

        dataloader_iter = iter(data_loader)
        running_loss = 0.
        batch_idx = 0
        total_loss = 0.0
        iter_times_list = list()
        h0 = model.init_hidden(config.training.batch_sz).to(device)

        # --- Report Validation Loss
        for input_tensor, output_tensor in dataloader_iter:
            if input_tensor.shape[0] != config.training.batch_sz:
                continue

            batch_idx += 1
            h = h0.data
            pos_pred, _ = model(x=input_tensor.to(device).float(), h=h)
            loss = criterion(pos_pred, output_tensor.to(device).float())
            optimizer.zero_grad()

            running_loss += loss.item()

            if batch_idx % print_every == 0:
                iter_time = time.time() - iter_start_time
                iter_times_list.append(iter_time)
                iter_start_time = time.time()
                print(f'[Validation] Epoch {epoch}', end='  ')
                print(f'Iter Time:{iter_time:.3f}', end='  ')
                print(f'Step:{batch_idx}/{len(data_loader):<8}', end='  ')
                print(f'l: {running_loss/batch_idx:<10}')

        total_loss = running_loss/len(data_loader)
        epoch_time_str = time.strftime("%H:%M:%S",
                                       time.gmtime(time.time() - epoch_start_time))
        print(f'Totals: Validation  loss: {total_loss}  time: {epoch_time_str}')
        print()

        # --- Report Error (L2 dist)
        error_tensor = torch.zeros(dataset.get_num_files())
        test_pbar = tqdm(list(range(dataset.get_num_files())))
        for sid in test_pbar:
            test_pbar.set_description(f'Sample Num: {sid}')
            sample_feats, sample_pos, _ = dataset.get_sample_by_index(sid)
            sample_feats = torch.Tensor(sample_feats).unsqueeze(0).to(device).float()
            sample_pos = torch.Tensor(sample_pos).to(device)
            pred_pos = model.infer(sample_feats).squeeze(0)

            seq_len = min(len(pred_pos), len(sample_pos))
            error_tensor[sid] = calc_seq_dist(pred_pos[:seq_len],
                                              sample_pos[:seq_len])

        total_error = error_tensor.mean().item()
        print(f'Totals: Validation error: {total_error:06f}')
        print()

        model_logger.val.add_scalar(loss_iter_str, total_loss, global_step)
        model_logger.test.add_scalar(loss_iter_str, total_error, global_step)
        
        if total_error < best.error:
            best.loss = total_loss
            best.error = total_error
            best.epoch = epoch
            if osp.exists(best.path):
                os.remove(best.path)
            best.path = osp.join(config.paths.save_dir,
                                 f'best_{epoch:02d}_{total_error:06f}.pt')
            save_checkpoint(epoch=epoch,
                            model=model,
                            model_params=model.params,
                            optimizer=optimizer,
                            optimizer_params=dict(lr=config.training.learning_rate),
                            loss=total_loss,
                            global_step=global_step,
                            save_path=best.path)
        else:
            if (epoch - best.epoch) >= config.training.early_stop:
                print(f'Early Stop @ Epoch {epoch}')
                print('Best model:')
                print(f'    Epoch:      {best.epoch}')
                print(f'    Val Loss:   {best.loss}')
                print(f'    Val Error:  {best.error}')
                print(f'    Path:       {best.path}')
                exit(0)

    return best


def run(model, optimizer, dataloaders, valid_dataset, config,
        start_epoch, log_start_step, device, print_every):

    model_logger = ModelLogger(config.paths.log_dir, f'test')
    criterion = get_criterion(config.training.loss)
    criterion.to(device)

    n_epochs = config.training.num_epochs
    start_time = time.time()
    global_step = log_start_step
    best = edict(loss=1e9, error=1e9, epoch=-1, path='best.pt')
    for epoch in range(start_epoch, start_epoch + n_epochs):
        print(f'Epoch {epoch}/{start_epoch + n_epochs - 1}')
        print('-' * 70)

        global_step = train_epoch(epoch,
                                  model,
                                  optimizer,
                                  criterion,
                                  dataloaders['train'],
                                  config,
                                  model_logger,
                                  global_step,
                                  device,
                                  print_every)
        best = validate_epoch(epoch,
                              model,
                              optimizer,
                              criterion,
                              dataloaders['valid'],
                              valid_dataset,
                              config,
                              model_logger,
                              global_step,
                              best,
                              device,
                              print_every)

    # --- Print total time
    total_time = time.time() - start_time
    total_train_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f'Total training time: {total_train_time_str}')


def main(local_config, config):
    # --- Init params
    start_epoch = 1
    log_start_step = 0

    # --- Build paths
    train_dataset_path = osp.join(local_config.datasets_dir,
                                  config.data.train_path)
    valid_dataset_path = osp.join(local_config.datasets_dir,
                                  config.data.valid_path)

    model_save_dir = osp.join(local_config.models_dir,
                              config.model.save_dir)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    log_dir = osp.join(local_config.logs_dir,
                       config.log.save_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    config['paths'] = dict()
    config.paths.save_dir = model_save_dir
    config.paths.log_dir = log_dir
    config.paths.train_dataset = train_dataset_path
    config.paths.valid_dataset = valid_dataset_path

    # --- Create torch device
    device = init_torch_device(args.gpu_id)

    # --- Create datasets
    print('Loading Training data')
    train_dataset = TongueMocapDataset(train_dataset_path,
                                       num_files=config.data.num_train_files,
                                       win_sz=config.data.win_sz,
                                       stride=config.data.win_stride,
                                       pose_only=True)
    print(f'Training samples:    {len(train_dataset)}')

    print('Loading Validation data')
    valid_dataset = TongueMocapDataset(valid_dataset_path,
                                       num_files=config.data.num_train_files,
                                       win_sz=config.data.win_sz,
                                       stride=config.data.win_stride,
                                       pose_only=True)
    print(f'Validation samples:  {len(valid_dataset)}')

    # --- Build data loaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.training.batch_sz,
                                  shuffle=True,
                                  num_workers=config.data.num_dataload_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.training.batch_sz,
                                  shuffle=False,
                                  num_workers=config.data.num_dataload_workers,
                                  pin_memory=True)
    dataloaders = dict(train=train_dataloader,
                       valid=valid_dataloader)

    # --- Build model from checkpoint or scratch
    if 'checkpoint' not in config:
        print('Building new model')
        rnn = GRU(config.model.input_sz,
                  config.model.hidden_sz,
                  config.model.output_sz,
                  n_layers=config.model.num_layers,
                  dropout=config.training.dropout,
                  bidirectional=config.model.bidir,
                  embedding_dim=config.model.embedding_sz)
        rnn.to(device)

        print('Building new optimizer')
        rnn_optimizer = optim.Adam(rnn.parameters(),
                                   lr=config.training.learning_rate)
    else:
        print('Loading model checkpoint')
        checkpoint_path = osp.join(local_config.models_dir,
                                   config.checkpoint.path)
        checkpoint = torch.load(checkpoint_path)
        rnn = GRU(**checkpoint['model_params'])
        rnn.load_state_dict(checkpoint['model_state_dict'])
        rnn.to(device)

        print('Loading optimizer checkpoint')
        rnn_optimizer = optim.Adam(rnn.parameters(),
                                   lr=checkpoint['optimizer_params']['lr'])
        rnn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = int(Path(checkpoint_path).stem) + 1
        log_start_step = checkpoint['global_step']

    print_training_header(rnn, config)

    run(rnn,
        rnn_optimizer,
        dataloaders,
        valid_dataset,
        config,
        start_epoch,
        log_start_step,
        device,
        print_every=200)


if __name__ == '__main__':
    # Parse input arguments
    args = parse_args()
    # Load configurations
    yaml = YAML(typ='safe')
    # -- machine configuration
    local_config = edict(yaml.load(open(args.local_config_path)))
    # -- training configuration
    config = edict(yaml.load(open(args.config_path)))

    main(local_config, config)
