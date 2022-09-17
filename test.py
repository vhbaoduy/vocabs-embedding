import argparse

import torch.optim.lr_scheduler
from trainer import *
from transforms import *
from torch.utils.data import DataLoader
from losses import *
from metrics import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-config_file', default='configs.yaml', type=str, help='name of config file')
    parser.add_argument('-model_name', default='resnet15', type=str,
                        choices=['resnet15', 'resnext'],
                        help='model name as backbone')
    parser.add_argument('-model_path', type=str, help='path to model')
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
    model_path = args.model_path
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
    df_test = pd.read_csv(os.path.join(dataset_cfgs['output_path'], 'test.csv'))
    test_transform = build_transform(audio_cfgs,
                                     mode='valid',
                                     feature_name=feature)
    test_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                         df_test,
                                         audio_cfgs['sample_rate'],
                                         labels=dataset_cfgs['labels'],
                                         transform=test_transform)

    test_dataloader = DataLoader(test_dataset,
                                 num_workers=param_cfgs['num_workers'],
                                 pin_memory=use_gpu)

    model = Network(model_name=model_name,
                    embedding_dims=embedding_dims,
                    l2_norm=param_cfgs['l2_normalized'],
                    num_classes=len(dataset_cfgs['labels']))

    model = torch.load(model_path)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    loss_fn = make_loss_fn(selective_type=triplet_selector,
                           margin=param_cfgs['triplet_margin'],
                           alpha=args.alpha,
                           beta=args.beta)

    metric_learnings = [AccumulatedAccuracyMetric(), AverageNonzeroTripletsMetric()]

    do_test(model=model,
            val_loader=test_dataloader,
            loss_fn=loss_fn,
            metrics=metric_learnings,
            use_gpu=use_gpu)