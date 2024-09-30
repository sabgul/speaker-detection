import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from src.ikrlib import wav16khz2mfcc, train_gmm, mfcc, logpdf_gmm
from src.data_preparator import VoiceDataPreparator

# _TRAIN_ITERATIONS = 64
# _GAUSSIANS_PER_CLASS = 8
# _HARD_SCORE_THRESHOLD: int = -500

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

    args.add_argument('--hard-decision-threshold', type=int, default=-500,
                      help='Decision threshold for determining the class data point belongs to.')
    args.add_argument('--gaussians-per-class', type=int, default=8,
                      help='Number of gaussians for each class.')
    args.add_argument('--train-iterations', type=int, default=64,
                      help='Max number of trainig iterations.')
    return args.parse_args()


def parse_filename(file_path):
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    person_id = parts[0]
    session_number = parts[1]
    return person_id, session_number


def load_or_compute_features(target_path, non_target_path):
    target_feats = {}
    non_target_feats = {}

    target_features = wav16khz2mfcc(target_path)
    for file_path in target_features:
        person_id, session_number = parse_filename(file_path)
        target_feats.setdefault(person_id, []).append(target_features[file_path])

    non_target_features = wav16khz2mfcc(non_target_path)
    for file_path in non_target_features:
        person_id, session_number = parse_filename(file_path)
        non_target_feats.setdefault(person_id, []).append(non_target_features[file_path])

    return target_feats, non_target_feats


def compute_gmm_parameters(features_dict):
    mean_values = {}
    covar_matrices = {}
    weights = {}

    for class_id, features_list in features_dict.items():
        class_features = np.concatenate(features_list, axis=0)  # Concatenate all arrays for the class

        covariance_matrices = []

        # equal weights for all Gaussians
        initial_weights = np.ones(_GAUSSIANS_PER_CLASS) / _GAUSSIANS_PER_CLASS

        # randomly select initial means for Gaussians
        initial_means_indices = np.random.choice(len(class_features), size=_GAUSSIANS_PER_CLASS, replace=False)
        initial_means = class_features[initial_means_indices]

        # Compute covariance matrices
        for i in range(_GAUSSIANS_PER_CLASS):
            covariance_matrices.append(np.cov(class_features.T))

        mean_values[class_id] = initial_means
        covar_matrices[class_id] = covariance_matrices
        weights[class_id] = initial_weights

    return mean_values, covar_matrices, weights


def train_gaussians(features_dict, weights, means, covars):
    updated_means = means
    updated_covars = covars
    updated_weights = weights

    running_tll_target = list()
    running_tll_non_target = list()

    for i in range(_TRAIN_ITERATIONS):
        print(f"Training iteration: {i}")
        for class_id, features_list in features_dict.items():
            class_features = np.concatenate(features_list, axis=0)
            updated_params = train_gmm(class_features, updated_weights[class_id],
                                       updated_means[class_id], updated_covars[class_id])

            updated_weights[class_id] = updated_params[0]
            updated_means[class_id] = updated_params[1]
            updated_covars[class_id] = updated_params[2]

            print(updated_params[-1])
            if class_id == 'm430':
                running_tll_target.append(updated_params[-1])
            elif class_id == 'non-target':
                running_tll_non_target.append(updated_params[-1])
            else:
                raise NotImplementedError

    # plt.plot(running_tll_target)
    # plt.plot(running_tll_non_target)
    # plt.show()

    return updated_weights, updated_means, updated_covars


def print_gmm_parameters(mean_values, covar_matrices, weights):
    for class_id, means in mean_values.items():
        print(f"Class ID: {class_id}")
        print("Mean Values:")
        for i, mean in enumerate(means):
            print(f"Gaussian {i + 1}: {mean}")
        print("Covariance Matrices:")
        for i, covar_matrix in enumerate(covar_matrices[class_id]):
            print(f"Gaussian {i + 1}: {covar_matrix}")
        print(f"Weights: {weights[class_id]}")
        print("\n")


def evaluate(target_path, non_target_path):
    eval_target = wav16khz2mfcc(target_path)
    eval_non_target = wav16khz2mfcc(non_target_path)

    eval_sample_cnt = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    ''' Target data: results should be ones. '''
    ll_vals = dict()
    for eval_sample in list(eval_target.values()):
        for class_id, features_list in feats.items():
            ll_vals[class_id] = sum(logpdf_gmm(eval_sample,
                                               trained_weights[class_id],
                                               trained_means[class_id],
                                               trained_covars[class_id]))
        soft_score = (ll_vals['m430']) - (ll_vals['non-target'])
        hard_score = 1 if soft_score > _HARD_SCORE_THRESHOLD else 0
        print(f'Target soft-score: {soft_score}, hard-score: {hard_score}')

        gt = 1  # Because these are target samples...
        eval_sample_cnt += 1

        if hard_score == 1:
            TP += 1
        if hard_score == 0:
            FN += 1

    ''' Non-Target data: should be zeros. '''
    ll_vals = dict()
    for eval_sample in list(eval_non_target.values()):
        for class_id, features_list in feats.items():
            ll_vals[class_id] = sum(logpdf_gmm(eval_sample,
                                               trained_weights[class_id],
                                               trained_means[class_id],
                                               trained_covars[class_id]))
        soft_score = (ll_vals['m430']) - (ll_vals['non-target'])
        hard_score = 1 if soft_score > _HARD_SCORE_THRESHOLD else 0
        print(f'Non-target soft-score: {soft_score}, hard-score: {hard_score}')

        gt = 0  # Because these are non-target samples...
        eval_sample_cnt += 1

        if hard_score == 0:
            TN += 1
        if hard_score == 1:
            FP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Hold-out train set results:'
          f'Total accuracy: {round(accuracy*100, 2)}%\n'
          f'Precision: {round(precision * 100, 2)}%\n'
          f'Recall: {round(recall * 100, 2)}%\n'
          f'F1 Score: {round(f1_score * 100, 2)}%\n'
          f'TP: {TP}\n'
          f'TN: {TN}\n'
          f'FP: {FP}\n'
          f'FN: {FN}.\n')

    # this line is for automatic processing with script.
    # make sure it is the last printed line (if you want to use the script)
    print(f"tp:{TP} tn:{TN} fp:{FP} fn:{FN} accuracy:{round(accuracy*100, 2)}")


def _evaluate_test_data():
    test_data = wav16khz2mfcc(_EVAL_DATA_PATH)
    ll_vals = dict()
    with open("voice_gmm_eval.txt", "w") as file:
        for eval_sample_name, eval_sample in list(test_data.items()):
            file_name = eval_sample_name.split("/")[-1]
            file_name_without_extension = file_name.split(".")[0]
            for class_id, features_list in feats.items():
                ll_vals[class_id] = sum(logpdf_gmm(eval_sample,
                                                   trained_weights[class_id],
                                                   trained_means[class_id],
                                                   trained_covars[class_id]))
            soft_score = (ll_vals['m430']) - (ll_vals['non-target'])
            hard_score = 1 if soft_score > _HARD_SCORE_THRESHOLD else 0
            result_line = f'{file_name_without_extension} {soft_score} {hard_score}\n'
            print(f'Evaluating sample {eval_sample_name}, decision: {hard_score}.')
            file.write(result_line)


if __name__ == '__main__':
    args = parse_args()

    _HARD_SCORE_THRESHOLD = args.hard_decision_threshold
    _TRAIN_ITERATIONS = args.train_iterations
    _GAUSSIANS_PER_CLASS = args.gaussians_per_class

    # ---- Voice data preprocessing
    voice_preparator = VoiceDataPreparator(args.target_train_path, args.non_target_train_path)
    target_data = voice_preparator.load_data('target')
    non_target_data = voice_preparator.load_data('non-target')

    if _APPLY_AUGMENTATION:
        # calculate weights for both classes, to balance dataset
        augmentation_w_target_num = int((1 - len(target_data)/(len(target_data) + len(non_target_data)))
                                        * (len(target_data) + len(non_target_data)))
        augmentation_w_non_target_num = int((1 - len(non_target_data)/(len(target_data) + len(non_target_data)))
                                            * (len(target_data) + len(non_target_data)))

        print(f'Generating {augmentation_w_target_num} augmented samples for target class...')
        voice_augmented_target_data = voice_preparator.augment_data(target_data,
                                                                    augmentation_num=augmentation_w_target_num)

        print(f'Generating {augmentation_w_non_target_num} augmented samples for non-target clas...')
        voice_augmented_non_target_data = voice_preparator.augment_data(non_target_data,
                                                                        augmentation_num=augmentation_w_non_target_num)

        target_feats = target_data + voice_augmented_target_data
        non_target_feats = non_target_data + voice_augmented_non_target_data
    else:
        target_feats = target_data
        non_target_feats = non_target_data

    for idx, t_f in enumerate(target_feats):
        t_f_key = list(t_f.keys())[0]
        t_f_val_new_val = mfcc(list(t_f.values())[0], 400, 240, 512, 16000, 23, 13)
        target_feats[idx] = {t_f_key: t_f_val_new_val}

    for idx, n_t_f in enumerate(non_target_feats):
        n_t_f_key = list(n_t_f.keys())[0]
        n_t_f_val_new_val = mfcc(list(n_t_f.values())[0], 400, 240, 512, 16000, 23, 13)
        non_target_feats[idx] = {n_t_f_key: n_t_f_val_new_val}

    num_target_classes = len(target_feats)
    num_non_target_classes = len(non_target_feats)

    print(f"Number of instances in target dataset: {num_target_classes}")
    print(f"Number of instances in non-target dataset: {num_non_target_classes}")

    target_feats_aggregated = defaultdict(list)
    non_target_feats_aggregated = defaultdict(list)
    extract_prefix = lambda key: key.split('_')[0][0:]

    for t_dict in target_feats:
        key = list(t_dict.keys())[0]
        value = list(t_dict.values())[0]
        prefix = extract_prefix(key)
        target_feats_aggregated[prefix].append(value)

    for n_t_dict in non_target_feats:
        key = list(n_t_dict.keys())[0]
        value = list(n_t_dict.values())[0]

        non_target_feats_aggregated['non-target'].append(value)  # generates one class for all non-targets

    feats = {**(target_feats_aggregated), **(non_target_feats_aggregated)}
    # ------------------------------------

    # ------------- Parameters and training
    mean_values, covar_matrices, weights = compute_gmm_parameters(feats)
    trained_weights, trained_means, trained_covars = train_gaussians(feats, weights, mean_values, covar_matrices)
    # ------------------------------------

    #  ------------- Evaluate on dev dataset
    evaluate(args.target_test_path, args.non_target_test_path)

    #  ------------- Evaluate on test dataset for submission, and generate results into file
    # _evaluate_test_data()
