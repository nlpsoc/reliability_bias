from typing import List, Tuple
from collections import defaultdict
import numpy as np


# Internal consistency
def cronbachs_alpha(scores):
    r"""
    Cronbach's Alpha: https://en.wikipedia.org/wiki/Cronbach%27s_alpha#Systematic_and_conventional_formula
    Measure the consistency of items.
    Inputs:
        scores: np.array shape of (number_of_items, number_of_elements)
    """
    within_item_var = scores.var(axis=1, ddof=1).sum()  # n_items
    btw_item_var = scores.sum(axis=0).var(ddof=1)
    k = scores.shape[0]
    alpha = k / (k - 1) * (1 - within_item_var / btw_item_var)
    return alpha


# Test-retest reliability and inter-rater consistency
def icc(scores, icc_type='icc2'):
    r"""
    ICC implementation from
    https://github.com/Mind-the-Pineapple/ICC/blob/master/ICC/icc.py
    Inputs:
        scores: np.array shape of (number_of_items, number_of_elements)
    """
    n_subjects, n_raters = scores.shape

    ss_total = np.var(scores, ddof=1) * (n_subjects * n_raters - 1)
    ms_r = np.var(np.mean(scores, axis=1), ddof=1) * n_raters
    ms_c = np.var(np.mean(scores, axis=0), ddof=1) * n_subjects
    ms_e = (ss_total - ms_r * (n_subjects - 1) - ms_c * (n_raters - 1)) / ((n_subjects - 1) * (n_raters - 1))

    icc_val = None
    if icc_type == 'icc2':
        icc_val = (ms_r - ms_e) / (ms_r + (n_raters - 1) * ms_e + (n_raters / n_subjects) * (ms_c - ms_e))
    elif icc_type == 'icc3':
        icc_val = (ms_r - ms_e) / (ms_r + (n_raters - 1) * ms_e)
    return icc_val


class ReliabilityEstimator:

    def __init__(self,
                 bias_scores: np.ndarray,
                 bias_measures: List[str],
                 target_words: List[str],
                 gender_base_pairs: List[Tuple[str]]):
        r"""
        Estimating the reliability of word embedding gender bias scores.
        Args:
            bias_scores: Bias scores to examine,
                numpy array shape of (n_bias_measures, n_target_words, n_gbps, n_embeds).
            bias_measures: Names of bias measures.
            target_words: List of target words.
            gender_base_pairs: List of gender base pairs (tuple).
        """
        # Check whether dimensions match
        assert (bias_scores.shape[:3] == (len(bias_measures), len(target_words), len(gender_base_pairs)))

        self.bias_scores = bias_scores
        self.bias_measures = bias_measures
        self.target_words = target_words
        self.gender_base_pairs = gender_base_pairs

    def retrieve_bias_scores_of_selected_words(self,
                                               selected_target_words: List[str] = None,
                                               selected_gender_base_pairs: List[List[str]] = None):
        bias_scores, target_words, gender_base_pairs = self.bias_scores, self.target_words, self.gender_base_pairs
        # Indexing bias scores to get target word list related parts
        if selected_target_words is not None:
            selected_target_word_idxs = [target_word_idx for target_word_idx, target_word in enumerate(self.target_words)
                                         if target_word in set(selected_target_words)]
            bias_scores = bias_scores[:, selected_target_word_idxs, :, :]
            target_words = selected_target_words
        if selected_gender_base_pairs is not None:
            selected_gender_base_pair_idxs = [
                gender_base_pair_idx for gender_base_pair_idx, gender_base_pair in enumerate(self.gender_base_pairs)
                if gender_base_pair in set(selected_gender_base_pairs)]
            bias_scores = bias_scores[:, :, selected_gender_base_pair_idxs :]
            gender_base_pairs = selected_gender_base_pairs
        return bias_scores, target_words, gender_base_pairs

    def test_retest_target_words(self,
                                 selected_target_words: List[str] = None,
                                 selected_gender_base_pairs: List[Tuple[str]] = None):
        # Indexing bias scores and build dict
        target_word_test_retest_reliability_dict = defaultdict(lambda: {})  # bias_measure: target_word: icc2
        bias_scores, target_words, gender_base_pairs = self.retrieve_bias_scores_of_selected_words(
            selected_target_words=selected_target_words, selected_gender_base_pairs=selected_gender_base_pairs)
        # Calculating test-retest reliability for each bias measure / target word
        for bias_measure_idx, bias_measure in enumerate(self.bias_measures):
            bias_scores_of_measure = bias_scores[bias_measure_idx]
            for target_word_idx, target_word in enumerate(target_words):
                bias_scores_of_measure_and_target_word = bias_scores_of_measure[target_word_idx]  # n_gbps, n_embeds
                target_word_test_retest_reliability_dict[bias_measure][target_word] = icc(
                    bias_scores_of_measure_and_target_word, icc_type='icc2')
        return dict(target_word_test_retest_reliability_dict)

    def test_retest_gender_base_pairs(self,
                                      selected_target_words: List[str] = None,
                                      selected_gender_base_pairs: List[Tuple[str]] = None):
        # Indexing bias scores and build dict
        gender_base_pair_test_retest_reliability_dict = defaultdict(lambda: {})  # bias_measure: target_word: icc2
        bias_scores, target_words, gender_base_pairs = self.retrieve_bias_scores_of_selected_words(
            selected_target_words=selected_target_words, selected_gender_base_pairs=selected_gender_base_pairs)
        # Calculating test-retest reliability for each bias measure / gender base pair
        for bias_measure_idx, bias_measure in enumerate(self.bias_measures):
            bias_scores_of_measure = bias_scores[bias_measure_idx]
            for gender_base_pair_idx, gender_base_pair in enumerate(gender_base_pairs):
                bias_scores_of_measure_and_gender_base_pair = bias_scores_of_measure[:, gender_base_pair_idx]  # n_target_words, n_embeds
                gender_base_pair_test_retest_reliability_dict[bias_measure][gender_base_pair] = icc(
                    bias_scores_of_measure_and_gender_base_pair, icc_type='icc2')
        return dict(gender_base_pair_test_retest_reliability_dict)

    def inter_rater_consistency_target_words(self,
                                             selected_target_words: List[str] = None,
                                             selected_gender_base_pairs: List[Tuple[str]] = None):
        # Indexing bias scores and build dict
        target_words_inter_rater_consistency_dict = {}  # target_word: icc3
        bias_scores, target_words, gender_base_pairs = self.retrieve_bias_scores_of_selected_words(
            selected_target_words=selected_target_words, selected_gender_base_pairs=selected_gender_base_pairs)
        # Calculating inter-rater consistency for each target word
        bias_scores = np.mean(bias_scores, axis=-1)
        for target_word_idx, target_word in enumerate(target_words):
            bias_scores_of_target_word = bias_scores[:, target_word_idx, :].T  # n_gbps, n_bias_measures
            target_words_inter_rater_consistency_dict[target_word] = icc(bias_scores_of_target_word, icc_type='icc3')
        return target_words_inter_rater_consistency_dict

    def inter_rater_consistency_gender_base_pairs(self,
                                                  selected_target_words: List[str] = None,
                                                  selected_gender_base_pairs: List[Tuple[str]] = None):
        # Indexing bias scores and build dict
        gender_base_pairs_inter_rater_consistency_dict = {}  # gender base pair: icc3
        bias_scores, target_words, gender_base_pairs = self.retrieve_bias_scores_of_selected_words(
            selected_target_words=selected_target_words, selected_gender_base_pairs=selected_gender_base_pairs)
        # Calculating inter-rater consistency for each gender base pair
        bias_scores = np.mean(bias_scores, axis=-1)
        for gender_base_pair_idx, gender_base_pair in enumerate(gender_base_pairs):
            bias_scores_of_target_word = bias_scores[:, :, gender_base_pair_idx].T  # n_target_words, n_bias_measures
            gender_base_pairs_inter_rater_consistency_dict[gender_base_pair] = icc(
                bias_scores_of_target_word, icc_type='icc3')
        return gender_base_pairs_inter_rater_consistency_dict

    def internal_consistency(self,
                             selected_target_words: List[str] = None,
                             selected_gender_base_pairs: List[Tuple[str]] = None):
        # Indexing bias scores and build dict
        internal_consistency_dict = {'gender base pairs': {}, 'target words': {}}  # Cronbach's alpha
        bias_scores, target_words, gender_base_pairs = self.retrieve_bias_scores_of_selected_words(
            selected_target_words=selected_target_words, selected_gender_base_pairs=selected_gender_base_pairs)
        bias_scores = np.mean(bias_scores, axis=-1)  # n_bias_measures, n_target_words, n_gbps
        # Calculating internal consistency
        for bias_measure_idx, bias_measure in enumerate(self.bias_measures):
            bias_scores_of_measure = bias_scores[bias_measure_idx]  # n_target_words, n_gbps
            internal_consistency_dict['target words'][bias_measure] = cronbachs_alpha(bias_scores_of_measure)
            internal_consistency_dict['gender base pairs'][bias_measure] = cronbachs_alpha(bias_scores_of_measure.T)
        return internal_consistency_dict

