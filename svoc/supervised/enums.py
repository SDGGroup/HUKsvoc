from enum import Enum

class SupervisedModel(str, Enum):
    ## Nota: Do not change the order! dentro il metodo find_supervised_matches si itera su questi modelli in questo ordine
    LOGREG = 'logreg'
    SVM = 'svm'
    NAIVE_BAYES = 'naive-bayes'