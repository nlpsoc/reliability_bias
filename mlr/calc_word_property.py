import os
import statistics
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from scipy.linalg import orthogonal_procrustes
from loguru import logger
import tqdm

import utils
from bias_measure import k_nearest_words
import paths
import data_loader as dl

CORPORA = ['wikitext', 'reddit_ask_science', 'reddit_ask_historians']


# embed properties
def get_nn_sim(vocab, embed_model):
    nearest_embed, query_embed = k_nearest_words(vocab, embed_model, k=1, return_query=True)
    nn_sim_vals = np.sum(query_embed * nearest_embed.squeeze(axis=1), axis=1)
    nn_sim_dict = {word: nn_sim_val for word, nn_sim_val in zip(vocab, nn_sim_vals)}
    return nn_sim_dict


def get_l2_norm(vocab, embed_model):
    return {word: np.linalg.norm(embed_model[word]) for word in vocab}


def average_multiple_embed(embed_property_dicts):
    return {word: np.mean([
        embed_property_dict[word] for embed_property_dict in embed_property_dicts])
        for word in embed_property_dicts[0]}


# corpus properties
def get_word_frequency(vocab_path):
    r"""word frequency"""
    return dl.load_word_frequency_dict(vocab_path)


# word properties
def get_most_common_pos(vocab):
    brown_tagged = brown.tagged_words(tagset='universal')
    word_dists = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in brown_tagged)
    most_freq_pos_dict = {word: word_dists[word].most_common(1)[0][0] for word in word_dists}
    return {word: most_freq_pos_dict[word] for word in vocab if word in most_freq_pos_dict}


def get_num_of_senses(vocab):
    return {word: len(wn.synsets(word)) for word in vocab if len(wn.synsets(word)) != 0}


def alignment_similarity(embed_model_0, embed_model_1, words):
    embed_matrix_0, embed_matrix_1 = zip(*[[embed_model_0[word], embed_model_1[word]]
                                           for word in set(embed_model_0) & set(embed_model_1)])
    align_matrix, _ = orthogonal_procrustes(np.array(embed_matrix_0), np.array(embed_matrix_1))
    cosine_similarity_dict = {}
    for word in words:
        word_embed_0 = embed_model_0[word].dot(align_matrix)
        word_embed_1 = embed_model_1[word]
        cosine_similarity = np.dot(utils.normalize_embed(word_embed_0), utils.normalize_embed(word_embed_1))
        cosine_similarity_dict[word] = float(cosine_similarity)
    return cosine_similarity_dict


def get_average_alignment_similarity(embed_models, words):
    orthogonal_procrustes_similarity_dicts = []
    calculated_idx = 0
    n_alignments = int(len(embed_models) * (len(embed_models) - 1) / 2)
    for embed_model_idx, embed_model_0 in enumerate(embed_models):
        logger.info(f'processing #{calculated_idx} alignment, {n_alignments} in total')
        calculated_idx += (len(embed_models) - embed_model_idx - 1)
        for embed_model_1 in embed_models[embed_model_idx + 1:]:
            orthogonal_procrustes_similarity_dict = alignment_similarity(
                embed_model_0, embed_model_1, words)
            orthogonal_procrustes_similarity_dicts.append(orthogonal_procrustes_similarity_dict)
    average_orthogonal_procrustes_similarity_dict = {word: statistics.mean(
        [similarity_dict[word] for similarity_dict in orthogonal_procrustes_similarity_dicts])
        for word in words}
    return average_orthogonal_procrustes_similarity_dict


def main():
    os.chdir('../../')
    os.makedirs(paths.property_dir, exist_ok=True)
    for corpus in CORPORA:
        vocab_path, sgns_embed_paths, glove_embed_paths = dl.get_embed_paths_from_folder(paths.embed_folders[corpus])
        vocab = dl.load_vocab(vocab_path)
        # remove artificial unknown tokens
        vocab.discard('<unk>')
        vocab.discard('<raw_unk>')

        # word properties
        logger.info('processing word properties')
        most_common_pos_dict = get_most_common_pos(vocab)
        n_senses_dict = get_num_of_senses(vocab)
        vocab = set(most_common_pos_dict) & set(n_senses_dict)

        # embedding properties
        logger.info('processing embedding properties: ES, NN Sim and L2 Norm.')
        # sgns
        sgns_nn_sims_dict = []
        sgns_l2_norm_dict = []
        sgns_embed_models = []
        logger.info('processing SGNS')
        for embed_path in tqdm.tqdm(sgns_embed_paths):
            embed_model = dl.load_gensim_sgns(embed_path)
            sgns_nn_sims_dict.append(get_nn_sim(vocab, embed_model))
            sgns_l2_norm_dict.append(get_l2_norm(vocab, embed_model))
            sgns_embed_models.append(embed_model)
        sgns_nn_sims_dict = average_multiple_embed(sgns_nn_sims_dict)
        sgns_l2_norm_dict = average_multiple_embed(sgns_l2_norm_dict)
        sgns_embed_stability = get_average_alignment_similarity(sgns_embed_models, list(vocab))
        # glove
        logger.info('processing GloVe')
        glove_nn_sims_dict = []
        glove_l2_norm_dict = []
        glove_embed_models = []
        for embed_path in tqdm.tqdm(glove_embed_paths):
            embed_model = dl.load_glove(embed_path, vocab_path=vocab_path)
            glove_nn_sims_dict.append(get_nn_sim(vocab, embed_model))
            glove_l2_norm_dict.append(get_l2_norm(vocab, embed_model))
        glove_nn_sims_dict = average_multiple_embed(glove_nn_sims_dict)
        glove_l2_norm_dict = average_multiple_embed(glove_l2_norm_dict)
        glove_embed_stability = get_average_alignment_similarity(glove_embed_models, list(vocab))

        # corpus properties
        logger.info('processing corpus properties: word frequency and dispersion')
        # word frequency
        frequency_dict = get_word_frequency(vocab_path)
        frequency_dict = {word: frequency_dict[word] for word in vocab}

        logger.info('saving to csv')
        words_with_properties = vocab & set(sgns_embed_stability)
        logger.info(f'valid words: {len(words_with_properties)}')
        property_dict = {
            'sgns_nn_sim': {word: sgns_nn_sims_dict[word] for word in words_with_properties},
            'glove_nn_sim': {word: glove_nn_sims_dict[word] for word in words_with_properties},
            'sgns_l2_norm': {word: sgns_l2_norm_dict[word] for word in words_with_properties},
            'glove_l2_norm': {word: glove_l2_norm_dict[word] for word in words_with_properties},
            'frequency': {word: frequency_dict[word] for word in words_with_properties},
            'most_common_pos': {word: most_common_pos_dict[word] for word in words_with_properties},
            'n_senses': {word: n_senses_dict[word] for word in words_with_properties},
            'sgns_es': {word: sgns_embed_stability[word] for word in words_with_properties},
            'glove_es': {word: glove_embed_stability[word] for word in words_with_properties},
        }
        property_df = pd.DataFrame(property_dict)
        property_csv_path = os.path.join(paths.property_dir, f'{corpus}.csv')
        property_df.to_csv(property_csv_path)


if __name__ == '__main__':
    main()
