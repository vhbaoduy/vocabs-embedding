from datasets import BackgroundNoiseDataset
from .transform_wav import *
from .transform_stft import *
from torchvision.transforms import Compose

def build_transform(audio_preprocessing_cfg,
                    mode='train',
                    feature_name='mel_spectrogram',
                    background_noise_path=None):
    """
    Build transform for train and valid dataset
    :param background_noise_path:
    :param feature_name:
    :param audio_preprocessing_cfg: configuration of audio preprocessing
    :param mode: 'train', 'valid'
    :return: data augmentation, background noise, feature transform
    """
    if mode == 'train':
        if feature_name == 'mel_spectrogram':
            data_aug_transform = Compose(
                [ChangeAmplitude(),
                 ChangeSpeedAndPitchAudio(),
                 FixAudioLength(),
                 ToSTFT(),
                 StretchAudioOnSTFT(),
                 TimeshiftAudioOnSTFT(),
                 FixSTFTDimension()
                 ])
            if background_noise_path is not None:
                bg_dataset = BackgroundNoiseDataset(background_noise_path, data_aug_transform,audio_preprocessing_cfg['sample_rate'])
                add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
            train_feature_transform = Compose(
                [ToMelSpectrogramFromSTFT(n_mels=audio_preprocessing_cfg['n_mels']),
                 DeleteSTFT(),
                 ToTensor('mel_spectrogram', 'input')])
        if feature_name == 'mfcc':
            data_aug_transform = Compose(
                [ChangeAmplitude(),
                 ChangeSpeedAndPitchAudio(),
                 FixAudioLength()])
            if background_noise_path is not None:
                bg_dataset = BackgroundNoiseDataset(background_noise_path, data_aug_transform,
                                                    audio_preprocessing_cfg['sample_rate'])
                add_bg_noise = AddBackgroundNoise(bg_dataset)
            train_feature_transform = Compose([
                ToMFCCs(n_mfcc=audio_preprocessing_cfg['n_mfcc']),
                ToTensor('mfcc_feature', 'input')
            ])

        return Compose([data_aug_transform, add_bg_noise, train_feature_transform])

    if mode == 'valid':
        if feature_name == 'mel_spectrogram':
            valid_transform = Compose([FixAudioLength(),
                                       ToMelSpectrogram(n_mels=audio_preprocessing_cfg['n_mels']),
                                       ToTensor('mel_spectrogram', 'input')])
        if feature_name == 'mfcc':
            valid_transform = Compose([
                FixAudioLength(),
                ToMFCCs(n_mfcc=audio_preprocessing_cfg['n_mfcc']),
                ToTensor('mfcc_feature', 'input')
            ])
        return valid_transform

    return None
