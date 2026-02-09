"""Enumerations for supervised matching models.

This module defines the types of supervised machine learning models
available for outlet matching in the SVOC system.
"""

from enum import Enum


class SupervisedModel(str, Enum):
    """Supervised machine learning models for outlet matching.
    
    This enum defines the available supervised models used for predicting
    whether two outlet records are a match. Each value corresponds to a
    specific trained model file.
    
    WARNING: Do not change the order of these values! The find_supervised_matches
    method iterates through these models in this specific order, and changing
    the order may affect matching behavior and results.
    
    Attributes:
        LOGREG: Logistic Regression model
        SVM: Support Vector Machine model
        NAIVE_BAYES: Naive Bayes classifier model
    """
    
    LOGREG = 'logreg'
    SVM = 'svm'
    NAIVE_BAYES = 'naive-bayes'