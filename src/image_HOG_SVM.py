import argparse
import typing

import numpy as np
from sklearn.svm import SVC
from src.ikrlib import png2fea
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import exposure
from data_preparator import FaceDataPreparator

_APPLY_AUGMENTATION: bool = True
_EVAL_DATA_PATH: str = '../data/eval/'


def parse_args() \
        -> argparse.Namespace:
    args = argparse.ArgumentParser()

    # evaluation data for implementation
    args.add_argument('--non-target-test-path', type=str, default='../data/non_target_dev/',
                      help='Path to folder of input non-target dataset.')
    args.add_argument('--target-test-path', type=str, default='../data/target_dev/',
                      help='Path to folder of input target dataset.')

    # training data
    args.add_argument('--non-target-train-path', type=str, default='../data/non_target_train/',
                      help='Path to folder of input non-target dataset.')
    args.add_argument('--target-train-path', type=str, default='../data/target_train/',
                      help='Path to folder of input target dataset.')
    return args.parse_args()


def get_hog(data):
    return [
        hog((image * 255).astype(np.uint8), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
            visualize=True, channel_axis=-1)[0]
        for image in data.values()]


def plot_hog(example_image: np.ndarray):
    fd, hog_image = hog(
        example_image,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1
    )

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(example_image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()


def obtain_augmentations(target_data_values, non_target_data_values):
    # ---- get weights for augmentations, to obtain balanced dataset
    augmentation_w_target_num = int(600 / 2 - len(target_data_values))

    augmentation_w_non_target_num = int(600 / 2 - len(non_target_data_values))

    # ----- generate augmentations
    print(f'Generating {augmentation_w_target_num} augmented samples for target class...')
    face_augmented_target_data = face_preparator.augment_data(target_data_values,
                                                              augmentation_num=augmentation_w_target_num)

    print(f'Generating {augmentation_w_non_target_num} augmented samples for non-target clas...')
    face_augmented_non_target_data = face_preparator.augment_data(non_target_data_values,
                                                                  augmentation_num=augmentation_w_non_target_num)

    return face_augmented_target_data, face_augmented_non_target_data


def get_statistics(gt, predicted):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for true_val, pred_val in zip(gt, predicted):
        if true_val == 1 and pred_val == 1:
            TP += 1
        elif true_val == 0 and pred_val == 0:
            TN += 1
        elif true_val == 0 and pred_val == 1:
            FP += 1
        elif true_val == 1 and pred_val == 0:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Hold-out train set results:\n'
          f'Total accuracy: {round(accuracy * 100, 2)}%\n'
          f'Precision: {round(precision * 100, 2)}%\n'
          f'Recall: {round(recall * 100, 2)}%\n'
          f'F1 Score: {round(f1_score * 100, 2)}%\n'
          f'TP: {TP}\n'
          f'TN: {TN}\n'
          f'FP: {FP}\n'
          f'FN: {FN}.\n')


def _evaluate_test_data():
    test_data = png2fea(_EVAL_DATA_PATH)
    file_paths = list(test_data.keys())
    for i, path in enumerate(file_paths):
        name = path.split("/")[-1]
        file_name_without_extension = name.split(".")[0]
        file_paths[i] = file_name_without_extension

    test_data_features = get_hog(test_data)
    test_data_features = np.asarray(test_data_features)

    class_predictions = classifier.predict(test_data_features)
    certainity = np.max(classifier.predict_proba(test_data_features), axis=1)
    with open("face_hog_svm_eval3.txt", "w") as f:
        for path, prediction, certainty_score in zip(file_paths, class_predictions, certainity):
            f.write(f"{path} {certainty_score} {prediction}\n")


if __name__ == '__main__':
    args = parse_args()
    face_preparator = FaceDataPreparator(args.target_train_path,
                                         args.non_target_train_path)

    # --- load data
    non_target_data = png2fea(args.non_target_train_path)
    target_data = png2fea(args.target_train_path)

    # -------
    feats = {'non-target': [], 'target': []}
    for val in non_target_data.values():
        feats['non-target'].append((val * 255).astype(np.uint8))

    for val in target_data.values():
        feats['target'].append((val * 255).astype(np.uint8))

    target_data_values = feats['target']
    non_target_data_values = feats['non-target']

    if _APPLY_AUGMENTATION:
        face_augmented_target_data, face_augmented_non_target_data = \
            obtain_augmentations(target_data_values, non_target_data_values)

        # ----- append augmentations to original data
        target_feats = target_data_values + face_augmented_target_data
        non_target_feats = non_target_data_values + face_augmented_non_target_data
    else:
        target_feats = target_data_values
        non_target_feats = non_target_data_values

    # ---------------- feature extraction
    print(f'Extracting features...')
    hog_features_target = [
        hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)[0]
        for
        image in target_feats]

    hog_features_non_target = [
        hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)[0]
        for
        image in non_target_feats]

    # ---------------- prepare input data for training
    train_data = np.asarray(hog_features_target + hog_features_non_target)
    labels = np.asarray([1] * len(hog_features_target) + [0] * len(hog_features_non_target))

    # ---------------- classification
    print(f'Training SVM...')
    classifier = SVC(probability=True)
    classifier.fit(train_data, labels)

    # ---------------- load and prepare data for validation
    val_target_data = png2fea(args.target_test_path)
    val_non_target_data = png2fea(args.non_target_test_path)

    val_hog_features_target = get_hog(val_target_data)
    val_hog_features_non_target = get_hog(val_non_target_data)

    val_data = np.asarray(val_hog_features_target + val_hog_features_non_target)
    val_labels = np.asarray([1] * len(val_hog_features_target) + [0] * len(val_hog_features_non_target))

    print(f'Obtaining validation results...')
    class_predictions = classifier.predict(val_data)
    get_statistics(val_labels, class_predictions)

    probabilities = classifier.predict_proba(val_data)
    max_probabilities = np.max(probabilities, axis=1)

    # ---------------- generate results on eval directory for submission
    # _evaluate_test_data()
