import pandas as pd
import recordlinkage as rl
from pathlib import Path
from svoc.utils import load_pickle, save_pickle


def train_supervised_model(
    supervised_model: str,
    train_set_matches_index: pd.Index,
    train_set_features: pd.DataFrame,
    save: bool = False,
    pickle_path: Path | None = None,
):

    if supervised_model == 'svm':
        mdl = rl.SVMClassifier()
    elif supervised_model == 'logreg':
        mdl = rl.LogisticRegressionClassifier()
    elif supervised_model == 'naive-bayes':
        mdl = rl.NaiveBayesClassifier(binarize=0.9)
    else:
        raise ValueError(f"Model not recognized: {supervised_model}")

    mdl.fit(train_set_features, train_set_matches_index)

    if save:
        if pickle_path is None:
            raise ValueError("pickle_path must be provided if save is True")

        save_pickle(mdl, pickle_path)

    return mdl

def predict_supervised(
    features: pd.DataFrame,
    pickle_path: Path | None = None,
    threshold: float = 0.5,
):
    if pickle_path is None:
        raise ValueError("pickle_path must be provided")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    mdl = load_pickle(pickle_path)

    matches = mdl.prob(features).reset_index(name="p")

    return matches[matches["p"] > threshold]