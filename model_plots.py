import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def flatten(x):
    return list(itertools.chain.from_iterable(x))


class ClassColors(object):
    Positive = '#ef8a62'
    Negative = '#67a9cf'
    Orange = '#FF7F0E'
    Blue = '#1f77b4'
    CB_Blue = '#006ba4'
    CB_Orange = '#FF8019'


def feature_coefficients(estimator, feature_names, class_names=None, n_features=20, figsize=(10, 5)):
    n_classes = estimator.coef_.shape[0]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    axes = flatten(axes)

    for class_label in xrange(n_classes):
        coefficients = estimator.coef_[class_label, :]
        feature_names = np.asarray(feature_names)

        if n_features:
            pos_indices = np.argsort(coefficients)[-n_features:][::-1]
            neg_indices = np.argsort(coefficients)[:n_features][::-1]
            coefficients = np.hstack((coefficients[pos_indices], coefficients[neg_indices]))
            coef_names = np.hstack((feature_names.take(pos_indices),
                                    feature_names.take(neg_indices)))
        else:
            coef_names = feature_names

        colors = [ClassColors.CB_Orange if c > 0 else ClassColors.CB_Blue for c in coefficients]

        axes[class_label].barh(-np.arange(coefficients.shape[0]), coefficients, color=colors)
        axes[class_label].xaxis.grid(False)
        plt.sca(axes[class_label])

        plt.yticks(-np.arange(coefficients.shape[0]) + 0.4, coef_names)
        if class_names is not None:
            plt.title(class_names[class_label])

    return plt.gca()
