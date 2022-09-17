import argparse

import torch.optim.lr_scheduler
from trainer import *
from transforms import *
from torch.utils.data import DataLoader
from losses import *
from metrics import *

def get_scheduler(param_cfgs, optimizer):
    scheduler_name = param_cfgs['lr_scheduler']
    if scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=param_cfgs['lr_scheduler_patience'],
            factor=param_cfgs['lr_scheduler_gamma']
        )

    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            gamma=param_cfgs['lr_scheduler_gamma'],
            last_epoch=-1
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=param_cfgs['T_max']
        )
    return scheduler, scheduler_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-config_file', default='configs.yaml', type=str, help='name of config file')
    parser.add_argument('-model_name', default='resnet15', type=str,
                        choices=['resnet15', 'resnext'],
                        help='model name as backbone')
    parser.add_argument('-embedding_dims', type=int, default=128, help="dimension of embeddings")
    parser.add_argument('-feature', type=str, default='mel_spectrogram', choices=['mfcc', 'mel_spectrogram'],
                        help="type of feature input")
    parser.add_argument('-triplet_selector', type=str, default='hardest',
                        choices=['hardest', 'random_hard', 'semi_hard'],
                        help="type of triplet selector")
    parser.add_argument('-alpha', type=float, default=0.5)
    parser.add_argument('-beta', type=float, default=0.5)

    args = parser.parse_args()

    # Parse arguments
    config_file = args.config_file
    model_name = args.model_name
    embedding_dims = args.embedding_dims
    feature = args.feature
    triplet_selector = args.triplet_selector

    # Load configs
    configs = utils.load_config_file(os.path.join('./configs', config_file))
    dataset_cfgs = configs['Dataset']
    audio_cfgs = configs['AudioProcessing']
    param_cfgs = configs['Parameters']
    checkpoint_cfgs = configs['Checkpoint']

    if not os.path.exists(dataset_cfgs['output_path']):
        # Prepare data
        data_preparing = DataPreparing(dataset_cfgs['root_dir'],
                                       dataset_cfgs['labels'],
                                       dataset_cfgs['output_path'])
        data_preparing.create_dataframe()

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    # Load dataframe
    df_train = pd.read_csv(os.path.join(dataset_cfgs['output_path'], 'train.csv'))
    df_valid = pd.read_csv(os.path.join(dataset_cfgs['output_path'], 'valid.csv'))

    background_noise_path = None
    if dataset_cfgs['add_noise']:
        background_noise_path = os.path.join(dataset_cfgs['root_dir'], dataset_cfgs['background_noise_path'])

    # Build transform
    train_transform = build_transform(audio_cfgs,
                                      mode='train',
                                      feature_name=feature,
                                      background_noise_path=background_noise_path)
    valid_transform = build_transform(audio_cfgs,
                                      mode='valid',
                                      feature_name=feature)

    train_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                          df_train,
                                          audio_cfgs['sample_rate'],
                                          labels=dataset_cfgs['labels'],
                                          transform=train_transform)

    valid_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                          df_valid,
                                          audio_cfgs['sample_rate'],
                                          labels=dataset_cfgs['labels'],
                                          transform=valid_transform)

    n_samples = param_cfgs['samples_per_class']
    n_classes = param_cfgs['classes_per_batch']
    batch_size = n_classes * n_samples

    train_sampler = BalancedBatchSampler(train_dataset.get_labels(), n_classes, n_samples)
    valid_sampler = BalancedBatchSampler(valid_dataset.get_labels(), n_classes, n_samples)

    # Data loader
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=param_cfgs['num_workers'],
                                  pin_memory=use_gpu)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_sampler=valid_sampler,
                                  num_workers=param_cfgs['num_workers'],
                                  pin_memory=use_gpu)

    model = Network(model_name = model_name,
                    embedding_dims=embedding_dims,
                    l2_norm=param_cfgs['l2_normalized'],
                    num_classes=len(dataset_cfgs['labels']))

    lr = param_cfgs['lr']
    weight_decay = param_cfgs['weight_decay']
    if param_cfgs['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

    schedulers = get_scheduler(param_cfgs, optimizer)

    loss_fn = make_loss_fn(selective_type=triplet_selector,
                           margin=param_cfgs['triplet_margin'],
                           alpha=args.alpha,
                           beta=args.beta)

    checkpoint_path = None
    if checkpoint_cfgs['resume']:
        print("Resuming a checkpoint '%s'" % checkpoint_cfgs['checkpoint_name'])
        checkpoint_path = os.path.join(checkpoint_cfgs['checkpoint_path'], checkpoint_cfgs['checkpoint_name'])

        # Create checkpoint path
    if not os.path.exists(checkpoint_cfgs['checkpoint_path']):
        os.mkdir(checkpoint_cfgs['checkpoint_path'])

    metric_learnings = [AccumulatedAccuracyMetric(),AverageNonzeroTripletsMetric()]

    fit(
        model=model,
        train_loader=train_dataloader,
        val_loader=valid_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        schedulers=schedulers,
        metrics=metric_learnings,
        max_epochs=param_cfgs['max_epochs'],
        start_epoch=0,
        checkpoint_path=checkpoint_path,
        save_path=checkpoint_cfgs['checkpoint_path'],
        use_gpu=use_gpu
    )