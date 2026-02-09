"""Enumerations for automatic matching distance methods.

This module defines the types of distance and similarity calculation methods
available for automatic outlet matching in the SVOC system.
"""

from enum import Enum


class DistanceMethod(str, Enum):
    """Distance and similarity calculation methods for outlet matching.
    
    This enum defines the available methods for calculating similarity or distance
    between text fields (outlet names, addresses, postcodes). These methods are used
    in automatic matching filters to identify potential outlet record matches.
    
    Attributes:
        COSINE: Cosine similarity based on TF-IDF or character n-grams
        JAROWINKLER: Jaro-Winkler distance, optimized for short strings
        LEVENSHTEIN: Levenshtein edit distance (number of single-character edits)
        QGRAM: Q-gram distance based on character n-gram overlap
        EXACT: Exact string match (case-sensitive equality)
        SUBSTRING: One string is a substring of the other
        WORDSMATCH: Word-level matching (comparing word sets)
    """
    
    COSINE = "cosine" 
    JAROWINKLER = "jarowinkler" 
    LEVENSHTEIN = "levenshtein" 
    QGRAM = "qgram" 
    EXACT = "exact" 
    SUBSTRING = "substring" 
    WORDSMATCH = "wordsmatch" 

