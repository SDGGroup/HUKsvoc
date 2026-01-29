import pandas as pd
import recordlinkage as rl
from pathlib import Path
from svoc.utils import load_pickle, save_pickle, concat_l
from svoc.supervised.enums import SupervisedModel
from svoc.settings import Settings
from svoc.automatic.models import Distance
from svoc.datapreparation import prepare_data, make_upper_str
from svoc.automatic.features import get_features

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
    pickle_path: Path | None = None,
    threshold: float = 0.5,
):

    if pickle_path is None:
        pickle_path = Settings().SUPERVISED_MODELS_PATHS[model]

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
    models_path_dict: dict[SupervisedModel, Path] | None = None
):
    
    remaining_features = features.set_index(["ID_1","ID_2"]).copy()
    
    all_matches_supervised_l = []

    for mdl in SupervisedModel:
        if remaining_features.empty:
            break   
        matches_supervised = predict_supervised(remaining_features, model=mdl, pickle_path=models_path_dict[mdl])
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

def train_all_models(
        df_input: pd.DataFrame,
        input_cols_id_benchmark: str,
        input_cols: dict,
        df_benchmark: pd.DataFrame,
        benchmark_cols: dict,
        distances: list[Distance], 
        block_col: str | None = None,
        window: int = 1,
        path_models: dict[SupervisedModel, str] | None = None,
):
    # Data Preparation
    df_benchmark_clean = prepare_data(
    df=df_benchmark, dict_cols=benchmark_cols)
    df_input_clean = prepare_data(
        df=df_input, dict_cols=input_cols)

    training_matches = make_upper_str(
        df_input[~df_input[input_cols_id_benchmark].isna()][[input_cols["ID"], input_cols_id_benchmark]]
        )    
    matched_indexes = training_matches[input_cols_id_benchmark].drop_duplicates().tolist()
    training_matches = (
        training_matches 
        .rename(columns={input_cols_id_benchmark: 'ID_1', input_cols["ID"]: 'ID_2'})
        .set_index(["ID_1","ID_2"])
        .index
    )

    training_features = get_features(
        distances, 
        df_x=df_benchmark_clean, 
        df_y=df_input_clean,
        block_col=block_col, 
        window=window
        )
    
    training_features = (training_features[training_features["ID_1"].isin(matched_indexes)]
                        .set_index(["ID_1","ID_2"]))
    
    models = {}
    for model in SupervisedModel:

        if path_models is None:
            mdl = train_supervised_model(
                supervised_model=model,
                train_set_matches_index=training_matches,
                train_set_features=training_features,
                save=False,
                pickle_path=None
            )
        else:
            mdl = train_supervised_model(
                supervised_model=model,
                train_set_matches_index=training_matches,
                train_set_features=training_features,
                save=True,
                pickle_path=path_models[model]
            )

        models[model] = mdl

    return models