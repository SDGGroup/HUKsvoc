import pandas as pd
import recordlinkage as rl
from pathlib import Path
from svoc.utils import load_pickle, save_pickle, concat_l
from svoc.supervised.enums import SupervisedModel
from svoc.settings import Settings

def train_supervised_model(
    supervised_model: SupervisedModel,
    train_set_matches_index: pd.Index,
    train_set_features: pd.DataFrame,
    save: bool = False,
    pickle_path: Path | None = None,
):

    if supervised_model == SupervisedModel.SVM:
        mdl = rl.SVMClassifier()
    elif supervised_model == SupervisedModel.LOGREG:
        mdl = rl.LogisticRegressionClassifier()
    elif supervised_model == SupervisedModel.NAIVE_BAYES:
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
    model: SupervisedModel,
   # pickle_path: Path | None = None,
    threshold: float = 0.5,
):

    pickle_path = Settings().SUPERVISED_MODEL_PATH[model]

    if pickle_path is None:
        raise ValueError("pickle_path must be provided")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    mdl = load_pickle(pickle_path)

    if mdl.__class__.__name__ == "SVMClassifier": 
        matches = pd.DataFrame(index = mdl.predict(features))
        matches["score"] = threshold
    else:
        matches = pd.DataFrame(mdl.prob(features), columns=['score'])

    matches["match_type"] = "supervised"
    matches["model"] = model.value

    return matches[matches["score"] >= threshold]

def find_supervised_matches(
    features: pd.DataFrame,
    block_col: str 
):
    block_col = block_col.lower()
    if block_col not in features.columns:
        raise ValueError(
            f"block_col '{block_col}' not found in df_benchmark columns: "
            f"{list(features.columns)}"
        )
    remaining_features = (features
                          .drop(columns=block_col)
                          .set_index(["ID_1","ID_2"])
                          .copy())
    
    all_matches_supervised_l = []

    for mdl in SupervisedModel:
        if remaining_features.empty:
            break
        matches_supervised = predict_supervised(remaining_features, model=mdl)
        remaining_features = remaining_features.loc[~remaining_features.index.isin(matches_supervised.index)]
        all_matches_supervised_l.append(matches_supervised.reset_index())

    if all([l.empty for l in all_matches_supervised_l]):
        all_matches_supervised = pd.DataFrame()
    else:
        all_matches_supervised = (features
                                .merge(
                                    concat_l(all_matches_supervised_l), 
                                    on=['ID_1','ID_2'], 
                                    how='inner'
                                    ))
        
    remaining_features = remaining_features.reset_index()

    return all_matches_supervised, remaining_features
    