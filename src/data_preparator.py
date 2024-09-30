import os
import argparse
import typing

import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from src.ikrlib import png2fea
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from scipy.signal import fftconvolve


SAMPL_RATE = 16000  # 16kHz


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--non-target-path', type=str, default='../data/non_target_train/',
                      help='Path to folder of input non-target dataset.')
    args.add_argument('--target-path', type=str, default='../data/target_train/',
                      help='Path to folder of input target dataset.')

    return args.parse_args()


class VoiceDataPreparator:
    """
    Class containing helper functions for processing the voice dataset.
    Provides functions for removing artefacts with no information value and
    functions for augmentation of data to artificially enlarge the dataset.
    """

    def __init__(self, target_path: str, non_target_path: str) -> None:
        self.target_path = target_path
        self.non_target_path = non_target_path

    def load_data(self, dataset_type: str) \
            -> list[dict]:
        assert dataset_type in ['target', 'non-target']
        path = self.target_path if dataset_type == 'target' else self.non_target_path

        data = []
        for filename in os.listdir(path):
            if filename.endswith('.wav'):
                print(f'Loading audio sample: {filename}')
                f = os.path.join(path, filename)

                rate, original_audio = wavfile.read(f)
                assert (rate == 16000)
                # trim the first 1.5 seconds (remove artefact with no information value)
                trimmed_audio = original_audio.copy()
                trimmed_audio = trimmed_audio[int(1.5 * SAMPL_RATE):]  # sampling rate of 16khz

                data.append({filename: trimmed_audio})

        return data

    @staticmethod
    def save_data(data: list, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        for i, augmented_audios in enumerate(data):
            for j, augmented_audio in enumerate(augmented_audios):
                print(f'Saving audio sample: [{i}_{j}]')

                filepath = os.path.join(output_path, f"augmented_{i}_{j}.wav")
                sf.write(filepath, augmented_audio, samplerate=SAMPL_RATE, subtype='PCM_16')

    def augment_data(self, data: typing.Union[list, list[dict]], augmentation_num: int = -1) -> list:
        augmented_data = list()

        if augmentation_num == -1:  # Old solution, deprecated; to use this, audio is directly the list of features
            for i, audio in enumerate(data):
                print(f'Augmenting audio sample: {i}')

                original_audio = audio.copy()

                white_noised = self.apply_white_noise(original_audio.values())
                pitch_shifted = self.apply_pitch_shift(original_audio)
                reverb = self.apply_reverb(original_audio)
                combined_augmentation = self.apply_combined_augmentation(original_audio)

                augmented_audios = [original_audio, white_noised, pitch_shifted, reverb, combined_augmentation]
                augmented_data.append(augmented_audios)
        else:
            for augmentation_step in range(augmentation_num):
                data_sample = data[augmentation_step % len(data)]
                original_audio = list(data_sample.values())[0].copy()
                original_name = list(data_sample.keys())[0]

                print(f'Augmenting audio sample: {augmentation_step % len(data)}, name {original_name}...')
                combined_augmentation = self.apply_combined_augmentation(original_audio)

                augmented_data.append({f'{original_name}_{augmentation_step}': combined_augmentation})

        return augmented_data

    @staticmethod
    def apply_white_noise(audio: np.ndarray,
                          factor: float = 0.005) \
            -> np.ndarray:
        print('Applying white noise...')
        return audio + factor * np.random.randn(len(audio))

    @staticmethod
    def apply_pitch_shift(audio: np.ndarray,
                          n_steps: float = 8.0) \
            -> np.ndarray:
        print('Applying pitch shift...')
        return librosa.effects.pitch_shift(audio.astype(float), sr=SAMPL_RATE, n_steps=n_steps)

    # implementation inspired by https://ursinus-cs472a-s2021.github.io/Modules/Module7/Video1
    @staticmethod
    def apply_reverb(audio: np.ndarray,
                     room_size: int = 100,
                     decay_factor: float = 0.5,
                     num_echoes: int = 5) \
            -> np.ndarray:
        print('Applying reverb...')
        delay_time = room_size / 343

        # the length of the impulse response
        T = int(delay_time * SAMPL_RATE)

        # the impulse response
        h = np.zeros((num_echoes + 1) * T + 1)
        h[0] = 1
        h[T::T] = decay_factor
        h = h * np.exp(-np.arange(len(h)) / (SAMPL_RATE / 2))

        # applying reverb using convolution
        audio_with_reverb = fftconvolve(audio, h, mode='full')[:len(audio)]

        return audio_with_reverb

    def apply_combined_augmentation(self, audio: np.ndarray) -> np.ndarray:
        '''
        It is preferable to apply strong augmentation, due to the limited size of the data at hand. Lower probability
        might introduce bias for a particular class which would have particular augmentation applied.
        '''
        augmented_audio = audio

        p_1 = np.random.random()
        if p_1 < 0.7:
            rand_factor = np.random.uniform(0.001, 0.01)
            augmented_audio = self.apply_white_noise(augmented_audio,
                                                     factor=rand_factor)

        p_2 = np.random.random()
        if p_2 < 0.7:
            rand_n_steps = np.random.uniform(2.0, 12.0)
            augmented_audio = self.apply_pitch_shift(augmented_audio,
                                                     n_steps=rand_n_steps)

        p_3 = np.random.random()
        if p_3 < 0.7:
            rand_room_size = np.random.uniform(30, 200)
            rand_decay_factor = np.random.uniform(0.3, 0.7)
            rand_num_echoes = int(np.random.uniform(3.0, 10.0))

            augmented_audio = self.apply_reverb(augmented_audio,
                                                room_size=rand_room_size,
                                                decay_factor=rand_decay_factor,
                                                num_echoes=rand_num_echoes)

        p_4 = np.random.random()
        if p_4 < 0.7:
            rand_rate = np.random.uniform(0.5, 2.0)
            augmented_audio = librosa.effects.time_stretch(augmented_audio.astype(float), rate=rand_rate)

        return augmented_audio


class FaceDataPreparator:
    """
    Class containing helper functions for preprocessing the face dataset.
    Provides functions to artificially increase size of dataset by augmenting data.
    """

    def __init__(self, target_path: str, non_target_path: str) -> None:
        self.target_path = target_path
        self.non_target_path = non_target_path

    def augment_data(self, data: typing.Union[list, list[dict]], augmentation_num: int = -1) -> list:
        augmented_data = list()

        for augmentation_step in range(augmentation_num):
            data_sample = data[augmentation_step % len(data)]
            original_img = data_sample.copy()

            print(f'Augmenting img sample: {augmentation_step % len(data)}.')
            combined_augmentation = self.apply_combined_augmentation(original_img)
            augmented_data.append(combined_augmentation)

        return augmented_data

    def apply_combined_augmentation(self, img: np.ndarray) -> np.ndarray:
        augmented_img = (img * 255).astype(np.uint8)

        p_1 = np.random.random()
        if p_1 < 0.5:
            rand_angle = int(np.random.uniform(-45, 45))
            augmented_img = self.apply_rotation(augmented_img, rand_angle)

        p_2 = np.random.random()
        if p_2 < 0.5:
            rand_scale_factor = np.random.uniform(0.7, 1.5)
            augmented_img = self.apply_scale(augmented_img, rand_scale_factor)

        p_3 = np.random.random()
        if p_3 < 0.5:
            augmented_img = self.add_noise(augmented_img)

        p_4 = np.random.random()
        if p_4 < 0.5:
            rand_hue_mul = np.random.uniform(0.9, 1.1)
            augmented_img = self.adjust_hue_saturation(augmented_img, rand_hue_mul)

        p_6 = np.random.random()
        if p_6 < 0.5:
            rand_severity = int(np.random.uniform(1, 5.0))
            augmented_img = self.adjust_brigthness(augmented_img, rand_severity)

        p_7 = np.random.random()
        if p_7 < 0.5:
            rand_blur_strength = np.random.uniform(0.0, 2.0)
            augmented_img = self.apply_blur(augmented_img, rand_blur_strength)

        return augmented_img

    @staticmethod
    def apply_rotation(image_features: np.ndarray, rotation_angle: int = 45) -> np.ndarray:
        seq = iaa.Sequential([
            iaa.Affine(rotate=rotation_angle)
        ])
        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features

    @staticmethod
    def apply_scale(image_features: np.ndarray, scale: float = 1.0) -> np.ndarray:
        seq = iaa.Sequential([
            iaa.Affine(scale=scale)
        ])
        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features

    @staticmethod
    def add_noise(image_features: np.ndarray) -> np.ndarray:
        seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(per_channel=True)
        ])
        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features

    @staticmethod
    def apply_blur(image_features: np.ndarray, sigma_range: float = 1.5):
        seq = iaa.Sequential([
            iaa.GaussianBlur(sigma=sigma_range)
        ])
        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features

    @staticmethod
    def adjust_hue_saturation(image_features: np.ndarray, mul_hue: float = 1.0) -> np.ndarray:
        seq = iaa.Sequential([
            iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        ])

        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features

    @staticmethod
    def saturate(image_features: np.ndarray, severity: int = 1):
        seq = iaa.Sequential([
            iaa.imgcorruptlike.Saturate(severity=severity)
        ])
        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features

    @staticmethod
    def adjust_brigthness(image_features: np.ndarray, severity: int = 1):
        seq = iaa.Sequential([
            iaa.imgcorruptlike.Brightness(severity=severity)
        ])
        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features

    @staticmethod
    def flip_horizontally(image_features: np.ndarray):
        seq = iaa.Sequential([
            iaa.Fliplr(1.0)
        ])
        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features

    @staticmethod
    def flip_vertically(image_features: np.ndarray):
        seq = iaa.Sequential([
            iaa.Flipud(1.0)
        ])
        augmented_image_features = seq(images=[image_features])[0]
        return augmented_image_features


if __name__ == '__main__':
    args = parse_args()

    # for fixed, reproducible generation across various experiments
    np.random.seed(42)

    # ---- Voice data preprocessing
    voice_preparator = VoiceDataPreparator(args.target_path, args.non_target_path)
    # target_data = voice_preparator.load_data('target')
    # non_target_data = voice_preparator.load_data('non-target')
    #
    # augmentation_w_target_num = int((1 - len(target_data)/(len(target_data) + len(non_target_data))) * (len(target_data) + len(non_target_data)))
    # augmentation_w_non_target_num = int((1 - len(non_target_data)/(len(target_data) + len(non_target_data))) * (len(target_data) + len(non_target_data)))
    #
    # print(f'Generating {augmentation_w_target_num} augmented samples for target class...')
    # voice_augmented_target_data = voice_preparator.augment_data(target_data,
    #                                                             augmentation_num=augmentation_w_target_num)
    #
    # print(f'Generating {augmentation_w_non_target_num} augmented samples for non-target clas...')
    # voice_augmented_non_target_data = voice_preparator.augment_data(non_target_data,
    #                                                                 augmentation_num=augmentation_w_non_target_num)

    # save data if needed
    # voice_preparator.save_data(voice_augmented_target_data, './data/voice_augmented_target/')
    # voice_preparator.save_data(voice_augmented_non_target_data, './data/voice_augmented_non_target/')

    # ---- Face data preprocessing
    face_preparator = FaceDataPreparator(args.target_path, args.non_target_path)
    non_target_data = png2fea(args.non_target_path)
    target_data = png2fea(args.target_path)

    feats = {'non-target': [], 'target': []}
    for val in non_target_data.values():
        feats['non-target'].append(val)

    for val in target_data.values():
        feats['target'].append(val)

    target_data_values = feats['target']
    non_target_data_values = feats['non-target']

    augmentation_w_target_num = int((1 - len(target_data_values) / (len(target_data_values) + len(non_target_data_values)))
                                    * (len(target_data_values) + len(non_target_data_values))) * 100
    augmentation_w_non_target_num = int((1 - len(non_target_data_values) / (len(target_data_values) + len(non_target_data_values)))
                                        * (len(target_data_values) + len(non_target_data_values))) * 100

    print(f'Generating {augmentation_w_target_num} augmented samples for target class...')
    face_augmented_target_data = face_preparator.augment_data(target_data_values,
                                                              augmentation_num=augmentation_w_target_num)

    print(f'Generating {augmentation_w_non_target_num} augmented samples for non-target clas...')
    face_augmented_non_target_data = face_preparator.augment_data(non_target_data_values,
                                                                  augmentation_num=augmentation_w_non_target_num)

    target_feats = target_data_values + face_augmented_target_data
    non_target_feats = non_target_data_values + face_augmented_non_target_data
