from transforms import *
from datasets import *
from models import *
from losses import *
from metrics import *

from torch.utils.data import WeightedRandomSampler, DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler
import torch
from tensorboardX import SummaryWriter
import argparse
import pandas as pd
import os
from tqdm import tqdm
import time

start_timestamp = int(time.time() * 1000)
start_epoch = 0
best_accuracy = 0
best_loss = 1e100
global_step = 0


def main():
    global best_accuracy, best_loss, global_step, start_epoch
    parser = argparse.ArgumentParser(description='Train classifer for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default='configs.yaml', help='the name of config file')
    parser.add_argument('--encoder_name', type=str, required=True, help="name of encoder")
    parser.add_argument('--encoder_path', type=str, required=True, help="path to encoder")
    parser.add_argument('--n_dims', type=int, default=128, help="dimension of embeddings")
    parser.add_argument('--n_maps', type=int, default=45, help="n maps if encoder is resnet15")
    parser.add_argument('--l2_norm', type=bool, default=True, help="l2 option")
    parser.add_argument('--feature', type=str, default='mel_spectrogram', choices=['mfcc', 'mel_spectrogram'],
                        help="type of feature input")

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    device = 'cpu'
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        device = 'cuda'

    path = os.path.join('./configs', args.config_file)
    feature = args.feature
    n_dims = args.n_dims

    configs = utils.load_config_file(path)
    dataset_cfgs = configs['Dataset']
    # Load dataframe
    df_train = pd.read_csv(os.path.join(dataset_cfgs['output_path'], 'train.csv'))
    df_valid = pd.read_csv(os.path.join(dataset_cfgs['output_path'], 'valid.csv'))

    audio_preprocessing_cfgs = configs['AudioPreprocessing']

    background_noise_path = None
    if dataset_cfgs['add_noise']:
        background_noise_path = os.path.join(dataset_cfgs['root_dir'], dataset_cfgs['background_noise_path'])

    # Build transform
    train_transform = build_transform(audio_preprocessing_cfgs, mode='train', feature_name=feature,
                                      background_noise_path=background_noise_path)
    valid_transform = build_transform(audio_preprocessing_cfgs, mode='valid', feature_name=feature)

    train_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                          df_train,
                                          audio_preprocessing_cfgs['sample_rate'],
                                          labels=dataset_cfgs['labels'],
                                          transform=train_transform)

    decoder_params = configs['DecoderParameters']
    valid_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                          df_valid,
                                          audio_preprocessing_cfgs['sample_rate'],
                                          labels=dataset_cfgs['labels'],
                                          transform=valid_transform)

    batch_size = decoder_params['batch_size']
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=decoder_params['num_workers'],
                                  pin_memory=use_gpu)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=decoder_params['num_workers'],
                                  pin_memory=use_gpu)

    hidden_sizes = decoder_params['hidden_sizes']
    classifer = Classifier(len(dataset_cfgs['labels']), emb_dims=n_dims, use_softmax=True, hidden_cfgs=hidden_sizes)

    # Init optimizer
    lr = decoder_params['lr']
    weight_decay = decoder_params['weight_decay']
    if decoder_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(classifer.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(classifer.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

    # Init learning rate scheduler
    scheduler_name = decoder_params['lr_scheduler']
    if scheduler_name == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  patience=decoder_params['lr_scheduler_patience'],
                                                                  factor=decoder_params['lr_scheduler_gamma'])
    elif scheduler_name == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decoder_params['lr_scheduler_step_size'],
                                                       gamma=decoder_params['lr_scheduler_gamma'],
                                                       last_epoch=start_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    encoder = get_model(args.encoder_name, n_dims, l2_normalized=args.l2_norm, n_maps=args.n_maps)
    if use_gpu:
        encoder = torch.nn.DataParallel(encoder).cuda()
        classifer = torch.nn.DataParallel(classifer).cuda()
    # Load encoder
    if os.path.isfile(args.encoder_path):
        print("Load encoder from path")
        state_dict = torch.load(args.encoder_path, map_location=torch.device(device))
        # state_dict = torch.nn.DataParallel(state_dict)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        #
        # for k, v in state_dict.items():
        #     if 'module' not in k:
        #         k = 'module.' + k
        #     else:
        #         k = k.replace('features.module.', 'module.features.')
        #     new_state_dict[k] = v

        encoder.load_state_dict(state_dict)
        encoder.float()

    freeze(encoder)

    checkpoint_cfgs = configs['Checkpoint']
    # Resuming mode
    if checkpoint_cfgs['resume']:
        print("Resuming a checkpoint '%s'" % checkpoint_cfgs['checkpoint_name'])
        checkpoint = torch.load(os.path.join(checkpoint_cfgs['checkpoint_path'], checkpoint_cfgs['checkpoint_name']),
                                map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

    # Create checkpoint path
    if not os.path.exists(checkpoint_cfgs['checkpoint_path']):
        os.mkdir(checkpoint_cfgs['checkpoint_path'])

    # Init name model and board
    name = 'encoder-%s_%s_%s_%s_%s' % (args.encoder_name, decoder_params['optimizer'], batch_size, feature, scheduler_name)
    writer = SummaryWriter(comment='_speech_commands_' + name)

    # Init criterion
    criterion = torch.nn.CrossEntropyLoss()


    checkpoint_cfgs = configs['Checkpoint']

    def train(epoch):
        global global_step
        phase = 'train'
        print(f'Epoch {epoch} - lr {utils.get_lr(optimizer)}')
        writer.add_scalar('%s/learning_rate' % phase, utils.get_lr(optimizer), epoch)

        classifer.train()
        running_loss = 0.0
        it = 0
        correct = 0
        total = 0

        pbar = tqdm(train_dataloader)
        for batch in pbar:
            inputs = batch['input']
            targets = batch['target']

            inputs = torch.autograd.Variable(inputs, requires_grad=True)
            targets = torch.autograd.Variable(targets, requires_grad=False)

            if use_gpu:
                inputs = inputs.to(device)
                targets = targets.to(device)

            embeddings = encoder(inputs)
            preds = classifer(embeddings)

            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            it += 1
            global_step += 1
            running_loss += loss.item()
            predicted = preds.data.max(1, keepdim=True)[1]

            correct += predicted.eq(targets.data.view_as(predicted)).sum()
            total += targets.size(0)

            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100 * correct / total)
            })

        accuracy = correct / total
        epoch_loss = running_loss / it
        writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    def valid(epoch):
        global best_accuracy, best_loss, global_step

        phase = 'valid'
        classifer.eval()  # Set model to evaluate mode

        running_loss = 0.0
        it = 0
        correct = 0
        total = 0

        pbar = tqdm(valid_dataloader)
        with torch.no_grad():
            for batch in pbar:
                inputs = batch['input']
                # inputs = torch.unsqueeze(inputs, 1)
                targets = batch['target']

                if use_gpu:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                # forward
                embeddings = encoder(inputs)
                preds = classifer(embeddings)
                loss = criterion(preds, targets)

                # statistics
                it += 1
                global_step += 1
                running_loss += loss.item()
                pred = preds.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).sum()
                total += targets.size(0)

                writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (running_loss / it),
                    'acc': "%.02f%%" % (100 * correct / total)
                })

        accuracy = correct / total
        epoch_loss = running_loss / it
        writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

        save_checkpoint = {
            'epoch': epoch,
            'step': global_step,
            'state_dict': classifer.state_dict(),
            'loss': epoch_loss,
            'accuracy': accuracy,
            'optimizer': optimizer.state_dict(),
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(save_checkpoint,
                       checkpoint_cfgs['checkpoint_path'] + '/' + 'best-loss-speech-commands-checkpoint-%s.pth' % name)
            torch.save(classifer, checkpoint_cfgs['checkpoint_path'] + '/' + 'best-loss.pth')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(save_checkpoint,
                       checkpoint_cfgs['checkpoint_path'] + '/' + 'best-acc-speech-commands-checkpoint-%s.pth' % name)
            torch.save(classifer, checkpoint_cfgs['checkpoint_path'] + '/' + 'best-acc.pth')

        torch.save(save_checkpoint, checkpoint_cfgs['checkpoint_path'] + '/' + 'last-speech-commands-checkpoint.pth')
        return epoch_loss

    print("Training ...")
    since = time.time()
    for epoch in range(start_epoch, decoder_params['max_epochs']):
        if scheduler_name == 'step':
            lr_scheduler.step()

        train(epoch)
        epoch_loss = valid(epoch)

        if scheduler_name == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)

        time_elapsed = time.time() - since
        time_str = 'Total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60,
                                                                         time_elapsed % 60)
        print("%s, Best accuracy: %.02f%%, best loss %f" % (time_str, 100 * best_accuracy, best_loss))
    print("Finished")


if __name__ == "__main__":
    main()
