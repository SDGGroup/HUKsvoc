"""Supervised machine learning matching module.

This module provides functionality for training and applying supervised machine learning
models to outlet matching. It includes:
- Training classifiers (SVM, Logistic Regression, Naive Bayes)
- Making predictions on potential matches
- Finding matches using multiple models in sequence
- Complete training pipeline for all models
"""

import json
import pandas as pd
import recordlinkage as rl
from recordlinkage.base import BaseClassifier
from pathlib import Path
import warnings

from tqdm import tqdm
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
    """Train a supervised classification model for outlet matching.
    
    Trains one of the available supervised models (SVM, Logistic Regression, or Naive Bayes)
    using labeled training data. The model learns to predict whether two outlet records
    represent the same entity based on their feature similarities.
    
    Args:
        supervised_model: Type of model to train (from SupervisedModel enum)
        train_set_matches_index: Index of known matching record pairs for training
        train_set_features: Feature DataFrame with computed similarities for record pairs
        save: If True, save the trained model to disk. Default: False
        pickle_path: Path to save the model. Required if save=True. Default: None
        
    Returns:
        Trained classifier model (recordlinkage classifier instance)
        
    Raises:
        ValueError: If supervised_model is not recognized or if save=True but pickle_path=None
        TypeError: If inputs are not of expected types
    """
    if not isinstance(supervised_model, SupervisedModel):
        raise TypeError(
            f"supervised_model must be a SupervisedModel enum, got {type(supervised_model).__name__}"
        )
    if not isinstance(train_set_matches_index, pd.Index):
        raise TypeError(
            f"train_set_matches_index must be a pandas Index, got {type(train_set_matches_index).__name__}"
        )
    if not isinstance(train_set_features, pd.DataFrame):
        raise TypeError(
            f"train_set_features must be a pandas DataFrame, got {type(train_set_features).__name__}"
        )

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
) -> pd.DataFrame:
    """Predict matches using a trained supervised model.
    
    Loads a trained supervised model and uses it to predict which record pairs
    are matches based on their computed features. Returns predictions that meet
    or exceed the specified probability threshold.
    
    Args:
        features: DataFrame containing computed features for record pairs
        model: Type of supervised model to use (from SupervisedModel enum)
        pickle_path: Path to the saved model file. If None, uses default from Settings. Default: None
        threshold: Minimum probability score for a match (0.0-1.0). Default: 0.5
        
    Returns:
        DataFrame with columns:
        - ID_1, ID_2: Index of matched record pairs
        - score: Match probability/confidence score
        - match_type: Always "supervised"
        - model: Name of the model used
        
    Raises:
        ValueError: If threshold is not between 0 and 1
        TypeError: If inputs are not of expected types
        FileNotFoundError: If model file doesn't exist at pickle_path
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            f"features must be a pandas DataFrame, got {type(features).__name__}"
        )
    if not isinstance(model, SupervisedModel):
        raise TypeError(
            f"model must be a SupervisedModel enum, got {type(model).__name__}"
        )

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find matches using multiple supervised models in sequence.
    
    Applies supervised models in order (LogReg, SVM, Naive Bayes) to identify matches.
    Once a record pair is classified as a match by one model, it's excluded from
    subsequent model predictions. This sequential approach prioritizes earlier models.
    
    Args:
        features: DataFrame containing computed features for all candidate record pairs
        models_path_dict: Dictionary mapping SupervisedModel to model file paths.
                         If None, uses default paths from Settings. Default: None
        
    Returns:
        tuple containing:
        - all_matches_supervised: DataFrame of all matched pairs with scores and model info
        - remaining_features: DataFrame of unmatched pairs for further processing
        
    Raises:
        TypeError: If inputs are not of expected types
        
    Note:
        Models are applied in the order defined in SupervisedModel enum.
        Do not change the enum order as it affects matching priority.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            f"features must be a pandas DataFrame, got {type(features).__name__}"
        )
    if models_path_dict is not None and not isinstance(models_path_dict, dict):
        raise TypeError(
            f"models_path_dict must be a dictionary, got {type(models_path_dict).__name__}"
        )
    
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
        input_cols: dict[str, str],
        df_benchmark: pd.DataFrame,
        benchmark_cols: dict[str, str],
        distances: list[Distance], 
        groups: dict | None = None, 
        block_col: str | None = None,
        window: int = 1,
        path_models: dict[SupervisedModel, str] | None = None,
) -> dict[SupervisedModel, BaseClassifier]:
    """Train all supervised models using labeled training data.
    
    Complete training pipeline that:
    1. Prepares and cleans input and benchmark data
    2. Extracts labeled matches from input data
    3. Computes features using blocking or clustering
    4. Trains all supervised models (LogReg, SVM, Naive Bayes)
    5. Optionally saves trained models to disk
    
    Args:
        df_input: Input DataFrame with training records (must include benchmark ID column)
        input_cols_id_benchmark: Column name in df_input containing benchmark IDs for labeled matches
        input_cols: Dictionary mapping standard column names to df_input column names
        df_benchmark: Benchmark DataFrame with reference records
        benchmark_cols: Dictionary mapping standard column names to df_benchmark column names
        distances: List of Distance objects defining features to compute
        groups: Optional dictionary for clustering-based matching {postcode: [neighbors]}.
               If None, uses blocking instead. Default: None
        block_col: Column name to use for blocking/on which the clustering is based. If None and groups is None, full index matching is used.Default: None
        window: Window size for blocking (number of sorted neighbors). Default: 1
        path_models: Dictionary mapping SupervisedModel to save paths.
                    If None, models are not saved. Default: None
        
    Returns:
        Dictionary mapping SupervisedModel enum values to trained classifier instances
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If required columns are missing or data is invalid
        
    Warnings:
        Issues UserWarning if groups is None, indicating blocking will be used instead of clustering
    """
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError(
            f"df_input must be a pandas DataFrame, got {type(df_input).__name__}"
        )
    if not isinstance(df_benchmark, pd.DataFrame):
        raise TypeError(
            f"df_benchmark must be a pandas DataFrame, got {type(df_benchmark).__name__}"
        )
    if not isinstance(input_cols, dict):
        raise TypeError(
            f"input_cols must be a dictionary, got {type(input_cols).__name__}"
        )
    if not isinstance(benchmark_cols, dict):
        raise TypeError(
            f"benchmark_cols must be a dictionary, got {type(benchmark_cols).__name__}"
        )
    if not isinstance(distances, list):
        raise TypeError(
            f"distances must be a list, got {type(distances).__name__}"
        )


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

    if groups is None:
        warnings.warn("""
                      groups parameter is None. 
                      The parameter must be provided for clustering-based matching.
                      Running model training with blocking instead of clustering.
                      """)


        training_features = get_features(
            distances, 
            df_x=df_benchmark_clean, 
            df_y=df_input_clean,
            block_col=block_col, 
            window=window
            )
    else:
        # Vectorized pair generation for clustering
        bench_groups = df_benchmark_clean.groupby(block_col).groups
        input_groups = df_input_clean.groupby(block_col).groups
        
        candidate_indices_l = []
        for pc, neighbors in groups.items():
            if pc not in bench_groups:
                continue
            
            bench_ids = bench_groups[pc]
            input_ids = []
            for neighbor in neighbors:
                if neighbor in input_groups:
                    input_ids.extend(input_groups[neighbor])
            
            if not input_ids:
                continue
            
            candidate_indices_l.append(pd.MultiIndex.from_product([bench_ids, input_ids]))
            
        if not candidate_indices_l:
            raise ValueError("No candidate pairs generated from clusters for training.")
             
        candidate_links = candidate_indices_l[0].append(candidate_indices_l[1:])
        candidate_links.names = ['ID_1', 'ID_2']
        candidate_links = candidate_links.drop_duplicates()
        
        training_features = get_features(distances, df_x=df_benchmark_clean, df_y=df_input_clean, candidate_links=candidate_links)


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