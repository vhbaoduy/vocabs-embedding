import argparse

from transforms import *
from torch.utils.data import DataLoader
from metrics import *
from models import *
from datasets import *
from tqdm import tqdm
import os
import numpy as np


def do_inference(model,
                 data_loader,
                 metric,
                 save_feature,
                 path,
                 labels,
                 mode,
                 use_gpu):
    with torch.no_grad():
        metric.reset()
        features = {}
        model.eval()
        pbar = tqdm(data_loader, desc='Test: ')
        for batch in pbar:
            inputs = batch['input']
            targets = batch['target']

            if use_gpu:
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')

            preds, feat = model(inputs)
            metric(preds, targets, None)

            ordered_dict = {metric.name(): metric.value()}
            pbar.set_postfix(ordered_dict)

            if preds is not None:
                prediction = preds.data.max(1, keepdim=True)[1]
                prediction = prediction.cpu().numpy().ravel()

            # Convert gpu to cpu
            # preds = preds.cpu().numpy().ravel()
            targets = targets.cpu().numpy().ravel()
            feat = feat.cpu().numpy()

            if save_feature:
                cur_batch_size = len(batch['path'])
                for i in range(cur_batch_size):
                    file_name = batch['path'][i]
                    if preds is not None:
                        name_class = utils.index_to_label(labels, prediction[i])
                    truth_class = utils.index_to_label(labels, targets[i])
                    if mode == 'truth' or mode == 'intersect':
                        folder = os.path.join(path, truth_class)
                    elif model == 'predict':
                        folder = os.path.join(path, name_class)

                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    audio_name = file_name.split('/')[1].split('.')[0]

                    if mode == 'intersect':
                        if name_class == truth_class:
                            np.save(os.path.join(folder, truth_class + "_" + audio_name + ".npy"), feat[i])
                    else:
                        np.save(os.path.join(folder, truth_class + "_" + audio_name + ".npy"), feat[i])

                    if mode == 'truth':
                        key = truth_class
                    elif mode == 'predict':
                        key = name_class
                    else:
                        if truth_class == name_class:
                            key = truth_class
                        else:
                            continue

                    if key in features:
                        features[key].append(feat[i])
                    else:
                        features[key] = [feat[i]]

        print("Calculating mean ...")
        for label in labels:
            folder = os.path.join(path, label)
            np.save(os.path.join(folder, label + '_mean.npy'),
                    np.mean(np.array(features[label]), axis=0))

        return metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-config_file', default='configs.yaml', type=str, help='name of config file')
    parser.add_argument('-model_name', default='resnet15', type=str,
                        choices=['resnet15', 'resnext', 'bc_resnet'],
                        help='model name as backbone')
    parser.add_argument('-model_path', type=str, help='path to model')
    parser.add_argument('-embedding_dims', type=int, default=128, help="dimension of embeddings")
    parser.add_argument('-feature', type=str, default='mel_spectrogram', choices=['mfcc', 'mel_spectrogram'],
                        help="type of feature input")
    parser.add_argument('-path_to_df', type=str, help='path to dataframe')
    parser.add_argument('-batch_size', type=int, default=128,
                        help="batch size for inference")
    parser.add_argument('-save_feature', type=bool, required=True,
                        help="optional")
    parser.add_argument('-path', type=str, default='./inferences',
                        help="path to store features")
    parser.add_argument('-mode', type=str, choices=['truth', 'predict', 'intersect'], help='use ground truth')
    args = parser.parse_args()

    # Parse arguments
    config_file = args.config_file
    model_name = args.model_name
    embedding_dims = args.embedding_dims
    feature = args.feature
    model_path = args.model_path
    save_feature = args.save_feature
    path = args.path
    batch_size = args.batch_size
    path_to_df = args.path_to_df
    mode = args.mode

    # Load configs
    configs = utils.load_config_file(os.path.join('./configs', config_file))
    dataset_cfgs = configs['Dataset']
    audio_cfgs = configs['AudioProcessing']
    param_cfgs = configs['Parameters']
    checkpoint_cfgs = configs['Checkpoint']

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    labels = dataset_cfgs['labels']
    # Load dataframe
    df_test = pd.read_csv(path_to_df)
    test_transform = build_transform(audio_cfgs,
                                     mode='valid',
                                     feature_name=feature)
    test_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                         df_test,
                                         audio_cfgs['sample_rate'],
                                         labels=dataset_cfgs['labels'],
                                         transform=test_transform)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=param_cfgs['num_workers'],
                                 pin_memory=use_gpu)

    model = Network(model_name=model_name,
                    embedding_dims=embedding_dims,
                    l2_norm=param_cfgs['l2_normalized'],
                    num_classes=len(dataset_cfgs['labels']))

    model = torch.load(model_path, map_location=torch.device('cpu'))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    metric_learning = AccumulatedAccuracyMetric()
    if not os.path.exists(path):
        os.mkdir(path)

    do_inference(model=model,
                 data_loader=test_dataloader,
                 metric=metric_learning,
                 save_feature=save_feature,
                 path=path,
                 labels=labels,
                 mode=mode,
                 use_gpu=use_gpu)
