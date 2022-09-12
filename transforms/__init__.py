from datasets import BackgroundNoiseDataset
from .transform_wav import *
from .transform_stft import *
from torchvision.transforms import Compose


def build_transform(audio_preprocessing_cfg, mode='train', background_noise_path=None):
    """
    Build transform for train and valid dataset
    :param backgroud_noise_path:
    :param audio_preprocessing_cfg: configuration of audio preprocessing
    :param mode: 'train', 'valid'
    :return: data augmentation, background noise, feature transform
    """
    if mode == 'train':
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
        return Compose([data_aug_transform, add_bg_noise, train_feature_transform])

    if mode == 'valid':
        valid_transform = Compose([FixAudioLength(),
                                   ToMelSpectrogram(n_mels=audio_preprocessing_cfg['n_mels']),
                                   ToTensor('mel_spectrogram', 'input')])
        return valid_transform

    return None
