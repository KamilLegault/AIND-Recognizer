import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("inf")
        best_model = None

        for num_of_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_of_components).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                score = (-2.0 * logL) + (model.n_features * np.log(len(self.sequences)))
                if score < best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        models = []
        for num_of_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_of_components)
                model.fit(self.X, self.lengths)
                score = model.score(self.X)
                models.append((score, model))
            except Exception as e:
                pass
        scores = [score for score, model in models]
        sum_log_L = sum(scores)
        best_score = float("-inf")
        num_models = len(models)

        best_model = None
        for log_L, model in models:
            score = log_L - (sum_log_L - log_L) / (num_models - 1.0)
            if score > best_score:
                best_model = model
                best_score = score
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n_splits = min(3, len(self.sequences))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        # here we split the date into training sets and testing sets
        split_data = [(train_idx, test_idx) for train_idx, test_idx in kf.split(self.sequences)]
        dict = {} # dictionary of components and models
        #this first loops builds the dictionary
        for num_of_components in range(self.min_n_components, self.max_n_components + 1):
            for train_index, test_index in split_data:
                model = self.base_model(num_of_components)
                if model is None:
                    break
                test_index = list(test_index)
                train_index = list(train_index)
                test_data, test_data_lengths = combine_sequences(test_index, self.sequences)
                train_data, train_data_lengths = combine_sequences(train_index, self.sequences)
                try:
                    model.fit(train_data, train_data_lengths)
                    score_model_list = dict.get(num_of_components, [])
                    score = model.score(test_data, test_data_lengths)
                    score_model_list.append((score, model))
                    dict[num_of_components] = score_model_list # we add a model to the dictionary
                except Exception as e:
                    pass
        components_list = []
        for num_of_components in range(self.min_n_components, self.max_n_components + 1):
            score_model_list = dict.get(num_of_components, None)
            if score_model_list is None:
                continue
            scores = [score for score, model in score_model_list]
            components_list.append((np.average(scores), num_of_components))
        # now return result
        if not len(components_list) == 0:
            return None
        # Sort the components by avg score in decreasing order, and return the best model
        _, num_of_components = sorted(components_list, key=lambda key: key[0], reverse=True)[0]
        best_model = self.base_model(num_of_components)
        best_model.fit(self.X, self.lengths)

        return best_model
