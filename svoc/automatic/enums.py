from enum import Enum

class DistanceMethod(str, Enum):
    COSINE = "cosine"
    JAROWINKLER = "jarowinkler"
    LEVENSHTEIN = "levenshtein"
    QGRAM = "qgram"
    EXACT = "exact"
    SUBSTRING = "substring"
    WORDSMATCH = "wordsmatch"

