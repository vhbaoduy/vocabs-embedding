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
            # Convert gpu to cpu
            # preds = preds.cpu().numpy().ravel()
            target = target.cpu().numpy().ravel()
            feat = feat.cpu().numpy()

            if save_feature:
                cur_batch_size = len(batch['path'])
                ploop = tqdm(cur_batch_size, desc="Save features: ")
                for i in range(ploop):
                    file_name = batch['path'][i]
                    name_class = utils.index_to_label(labels, targets[i])
                    folder = os.path.join(path, name_class)
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    audio_name = file_name.split('\\')[1].split('.')[0]
                    # print(feats[i])
                    # print(feats[i].shape)
                    np.save(os.path.join(folder, audio_name + ".npy"), feat[i])
                    if features.has_key(name_class):
                        features[name_class].append(feat[i])
                    else:
                        features[name_class] = [feat[i]]

            pbar.set_postfix(ordered_dict)

        print("Calculating mean ...")
        for label in labels:
            folder = os.path.join(path, name_class)
            np.save(os.path.join(folder, label + '_mean.npy'),
                    np.mean(np.array(features[label])), axis=0)

        return metric


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
    parser.add_argument('-path_to_df', type=str, help='path to dataframe')
    parser.add_argument('-batch_size', type=int, default=128,
                        help="batch size for inference")
    parser.add_argument('-save_feature', type=bool, required=True,
                        help="optional")
    parser.add_argument('-path', type=str, default='./inferences',
                        help="path to store features")
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

    do_inference(model=model,
                 data_loader=test_dataloader,
                 metric=metric_learning,
                 save_feature=save_feature,
                 path=path,
                 labels=labels,
                 use_gpu=use_gpu)
