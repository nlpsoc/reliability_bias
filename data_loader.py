from typing import Dict
import string
import os

import numpy as np

import paths
import utils
import data.word_lists.weat_words as weat_words


def load_nips_16_gbp():
    return [(m.lower(), f.lower()) for (f, m) in utils.load_json(paths.nips_16_gbp_path)]


def load_pnas_18_gbp():
    return [(m.lower(), f.lower()) for (m, f) in utils.load_json(paths.pnas_18_gbp_path)]


def load_gbp():
    return list(set(load_pnas_18_gbp() + load_nips_16_gbp()))


def load_nips16_professions():
    return utils.load_json(paths.nips_16_profession_list_path)


def load_pnas18_professions():
    return utils.load_txt_list(paths.pnas_18_occupation_list_path)


def load_weat_gender_related_concepts():
    weat_concept_words = weat_words.concept_words
    concepts = ['career', 'family', 'math', 'arts', 'science', 'arts_2']
    concept2words = {}
    for concept in concepts:
        concept2words[concept] = weat_concept_words[concept]
    return concept2words


def load_pnas18_adjectives():
    return utils.load_txt_list(paths.pnas_18_adjective_list_path)


def load_google_10k():
    return utils.load_txt_list(paths.google_10k_most_frequent_list_path)


def load_gensim_sgns(embed_path) -> Dict[str, np.array]:
    # gensim word2vec model
    # embed_model.vocab: {word: (count, index)}
    from gensim.models import Word2Vec

    artificial_tokens = ['<unk>', '<raw_unk>']
    embed_model = Word2Vec.load(embed_path).wv
    embed_model = {word: embed_model[word] for word in embed_model.vocab if word not in artificial_tokens}
    return embed_model


def load_glove(embed_path, vocab_path=None):
    embed_model = {}
    f_embed = open(embed_path)
    f_vocab = open(vocab_path)
    for line_embed, line_vocab in zip(f_embed, f_vocab):
        try:
            word, _ = line_vocab.strip().split()
        except ValueError:  # ignore blanks
            continue
        line_embed = line_embed.strip().split()
        if len(line_embed) != 301:
            continue
        vector = np.array(line_embed[1:]).astype('float32')
        embed_model[word] = vector

    artificial_tokens = ['<unk>', '<raw_unk>']
    for token in artificial_tokens:
        embed_model.pop(token, None)

    return embed_model


def filter_word(word):
    """https://github.com/gonenhila/gender_bias_lipstick/blob/master/source/remaining_bias_2016.ipynb"""

    def has_punct(w):
        if any([c in string.punctuation for c in w]):
            return True
        return False

    def has_digit(w):
        if any([c in '0123456789' for c in w]):
            return True
        return False

    if word.lower() != word:
        return False
    if len(word) >= 20:
        return False
    if has_digit(word):
        return False
    if '_' in word:
        p = [has_punct(sub_w) for sub_w in word.split('_')]
        if any(p):
            return False
    if has_punct(word):
        return False

    return True


def retrieve_top_k_words_from_vocab(filename, k=50000):
    vocab = {}
    with open(filename) as fin:
        for line_idx, line in enumerate(fin):
            if line_idx >= k:
                break
            try:
                word, count = line.strip().split()
            except ValueError:  # ignore blanks
                continue
            if not filter_word(word):
                continue
            vocab[word] = int(count)
    return vocab


def load_word_frequency_dict(filename):
    word_frequency_dict = {}
    with open(filename) as fin:
        for line in fin:
            try:
                word, count = line.strip().split()
            except ValueError:  # ignore blanks
                continue
            word_frequency_dict[word] = count
    return word_frequency_dict


def load_vocab(filename):
    vocab = []
    with open(filename) as fin:
        for line in fin:
            try:
                word, _ = line.strip().split()
            except ValueError:  # ignore blanks
                continue
            vocab.append(word)
    return set(vocab)


def get_embed_paths_from_folder(folder_dir):
    vocab_path = os.path.join(folder_dir, 'vocab.txt')

    sgns_dir = os.path.join(folder_dir, 'sgns/')
    sgns_embed_paths = [os.path.join(sgns_dir, filename)
                        for filename in os.listdir(sgns_dir) if filename.endswith('.model')]

    glove_dir = os.path.join(folder_dir, 'glove/')
    glove_embed_paths = [os.path.join(glove_dir, filename)
                         for filename in os.listdir(glove_dir) if filename.endswith('.txt')]

    return vocab_path, sgns_embed_paths, glove_embed_paths
