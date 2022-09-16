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


def get_lr(opt):
    return opt.param_groups[0]['lr']


def main():
    global best_accuracy, global_step, start_epoch, start_timestamp, best_loss
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_dims', type=int, default=128, help="dimension of embeddings")
    parser.add_argument('--feature', type=str, default='mel_spectrogram', choices=['mfcc', 'mel_spectrogram'],
                        help="type of feature input")
    parser.add_argument('--config_file', type=str, default='configs.yaml',help="name of config file")
    parser.add_argument('--triplet_selector', type=str, default='hardest', choices=['hardest', 'random_hard','semi_hard'],
                        help="type of triplet selector")
    parser.add_argument('--model', type=str, default='resnet15', help="name of model")

    use_gpu = torch.cuda.is_available()
    device = 'cpu'
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        device = 'cuda'

    # Parse agr and load config
    args = parser.parse_args()
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

    valid_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                          df_valid,
                                          audio_preprocessing_cfgs['sample_rate'],
                                          labels=dataset_cfgs['labels'],
                                          transform=valid_transform)
    # print(len(dataset_cfgs['labels']))
    encoder_params = configs['EncoderParameters']

    # weights = train_dataset.make_weights_for_balanced_classes()
    # sampler = WeightedRandomSampler(weights, len(weights))

    n_samples = encoder_params['samples_per_class']
    n_classes = encoder_params['classes_per_batch']
    batch_size = n_classes * n_samples

    train_sampler = BalancedBatchSampler(train_dataset.get_labels(), n_classes, n_samples)
    valid_sampler = BalancedBatchSampler(valid_dataset.get_labels(), n_classes, n_samples)

    # Data loader
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=encoder_params['num_workers'],
                                  pin_memory=use_gpu)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_sampler=valid_sampler,
                                  num_workers=encoder_params['num_workers'],
                                  pin_memory=use_gpu)

    # Create model
    model = get_model(args.model, n_dims, encoder_params['use_l2_normalized'],n_maps=encoder_params['n_maps'])

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # Triplet Selector
    margin = encoder_params['margin']
    triplet_selector = RandomNegativeTripletSelector(margin)

    # Init criterion
    triplet_loss = OnlineTripletLoss(margin, triplet_selector)



    # Init optimizer
    lr = encoder_params['lr']
    weight_decay = encoder_params['weight_decay']
    if encoder_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

    # Init learning rate scheduler
    scheduler_name = encoder_params['lr_scheduler']
    if  scheduler_name == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  patience=encoder_params['lr_scheduler_patience'],
                                                                  factor=encoder_params['lr_scheduler_gamma'])
    elif scheduler_name == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=encoder_params['lr_scheduler_step_size'],
                                                       gamma=encoder_params['lr_scheduler_gamma'],
                                                       last_epoch=start_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 10)

    # Init metric
    metric_learning = AverageNonzeroTripletsMetric()

    checkpoint_cfgs = configs['Checkpoint']
    # Resuming mode
    if checkpoint_cfgs['resume']:
        print("Resuming a checkpoint '%s'" % checkpoint_cfgs['checkpoint_name'])
        checkpoint = torch.load(os.path.join(checkpoint_cfgs['checkpoint_path'], checkpoint_cfgs['checkpoint_name']))
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        # best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

    # Create checkpoint path
    if not os.path.exists(checkpoint_cfgs['checkpoint_path']):
        os.mkdir(checkpoint_cfgs['checkpoint_path'])

    # Init name model and board
    name = '%s_%s_%s_%s_%s' % (args.model, encoder_params['optimizer'], batch_size,feature,scheduler_name)
    writer = SummaryWriter(comment='_speech_commands_' + name)

    def train(epoch):
        global global_step
        # global metric_learning
        metric_learning.reset()

        phase = 'train'
        print(f'Epoch {epoch} - lr {get_lr(optimizer)}')
        writer.add_scalar('%s/learning_rate' % phase, get_lr(optimizer), epoch)

        model.train()
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

            embeddings = model(inputs)

            loss, total_triplet = triplet_loss(embeddings, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_learning(embeddings, targets, total_triplet)

            it += 1
            global_step += 1
            running_loss += loss.item()
            total += targets.size(0)

            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                metric_learning.name(): metric_learning.value()
                # 'acc': "%.02f%%" % (100 * correct / total)
            })

        epoch_loss = running_loss / it
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    def valid(epoch):
        global best_accuracy, best_loss, global_step
        # global metric_learning
        metric_learning.reset()

        phase = 'valid'
        model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        it = 0

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
                embeddings = model(inputs)
                loss, total_triplet = triplet_loss(embeddings, targets)

                metric_learning(embeddings, targets, total_triplet)

                # statistics
                it += 1
                global_step += 1
                running_loss += loss.item()

                writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (running_loss / it),
                    metric_learning.name(): metric_learning.value()
                })

        # accuracy = correct / total
        epoch_loss = running_loss / it
        # writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

        save_checkpoint = {
            'epoch': epoch,
            'step': global_step,
            'state_dict': model.state_dict(),
            'loss': epoch_loss,
            'optimizer': optimizer.state_dict(),
        }

        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     torch.save(save_checkpoint,
        #                configs.checkpoint_path + '/' + 'best-loss-speech-commands-checkpoint-%s.pth' % name)
        #     torch.save(model, configs.checkpoint_path + '/' + 'best-loss.pth')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(save_checkpoint,
                       checkpoint_cfgs['checkpoint_path'] + '/' + 'best-loss-%s-%s.pth' % (best_loss, name))
            # torch.save(model, configs.checkpoint_path + '/' + 'best-.pth')

        torch.save(save_checkpoint, checkpoint_cfgs['checkpoint_path'] + '/' + 'last-checkpoint.pth')
        return epoch_loss

    print("Training ...")
    since = time.time()
    for epoch in range(start_epoch, encoder_params['max_epochs']):
        if scheduler_name == 'step':
            lr_scheduler.step()

        train(epoch)
        epoch_loss = valid(epoch)

        if scheduler_name == 'cosine':
            lr_scheduler.step()

        if scheduler_name == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)

        time_elapsed = time.time() - since
        time_str = 'Total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60,
                                                                         time_elapsed % 60)
        print("%s, Best loss %f" % (time_str, best_loss))
    print("Finished")


if __name__ == '__main__':
    main()
