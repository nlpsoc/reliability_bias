"""
Word embedding gender bias measurements
1. The k-nearest algorithm is adopted from
    https://github.com/laura-burdick/embeddingStability/blob/master/stability.py
2. Bias measurements mainly follow AACL 2020 paper
"Robustness and Reliability of Gender Bias Assessment in Word Embeddings: The Role of Base Pairs"
Some codes are borrowed from https://github.com/alisonsneyd/Gender_bias_word_embeddings
"""
import numpy as np
import faiss
import utils


def batch_db_wa_bias(target_words, gender_base_pairs, embed_model):
    # gender base pairs embeddings
    male_embeds, female_embeds = zip(*[[embed_model[male_word], embed_model[female_word]]
                                       for [male_word, female_word] in gender_base_pairs])
    male_embeds = np.array([utils.normalize_embed(male_embed) for male_embed in male_embeds])
    female_embeds = np.array([utils.normalize_embed(female_embed) for female_embed in female_embeds])
    gender_base_pairs_embeds = male_embeds - female_embeds  # n_pairs, embed_dim
    # target words embeddings
    target_embeds = np.array([utils.normalize_embed(embed_model[target_word])
                              for target_word in target_words])  # n_target_words, embed_dim
    # bias scores
    bias_scores = np.matmul(target_embeds, gender_base_pairs_embeds.transpose())  # n_target_words, n_pairs
    return bias_scores


def batch_ripa_bias(target_words, gender_base_pairs, embed_model):
    # gender base pairs embeddings
    gender_base_pairs_embeds = np.array([
        utils.normalize_embed(embed_model[male_word] - embed_model[female_word])
        for [male_word, female_word] in gender_base_pairs])  # n_pairs, embed_dim
    # target words embeddings
    target_embeds = np.array([utils.normalize_embed(embed_model[target_word])
                              for target_word in target_words])  # n_target_words, embed_dim
    # bias scores
    bias_scores = np.matmul(target_embeds, gender_base_pairs_embeds.transpose())  # n_target_words, n_pairs
    return bias_scores


# k-nearest neighbors
def build_faiss_index(embed_model):
    # vocab: tuple of words
    # embed: tuple of embeddings of size (embed_dim, )
    vocab, embed = zip(*[[word, utils.normalize_embed(embed_model[word]).astype('float32')]
                         for word in embed_model])
    embed = np.array(embed)  # n_words, embed_dim
    searcher = faiss.IndexFlatL2(embed.shape[1])
    searcher.add(embed)
    return embed, vocab, searcher


def k_nearest_words(query_words, embed_model, k=100, return_query=False):
    embed, _, searcher = build_faiss_index(embed_model)
    query_embeds = np.array([utils.normalize_embed(
        embed_model[word]).astype('float32') for word in query_words])  # n_queries, embed_dim
    _, nearest_ids = searcher.search(query_embeds, k+1)
    nearest_ids = nearest_ids[:, 1:]  # n_queries, k
    # index embed (n_words, embed_dim) with nearest ids (n_queries, k) => n_queries, k, embed_dim
    top_k_embed = embed[nearest_ids]
    if return_query:
        return top_k_embed, query_embeds
    else:
        return top_k_embed


def batch_nbm_bias(target_words, gender_base_pairs, embed_model, k=100):
    gender_base_pairs_embeds = np.array([
        utils.normalize_embed(embed_model[male_word]) - utils.normalize_embed(embed_model[female_word])
        for [male_word, female_word] in gender_base_pairs])  # n_pairs, embed_dim
    nearest_neighbors_embed = k_nearest_words(target_words, embed_model, k=k)  # n_target_words, k, embed_dim
    db_wa_bias_scores = np.matmul(nearest_neighbors_embed, gender_base_pairs_embeds.T)  # n_target_words, k, n_pairs
    bias_scores = np.where(db_wa_bias_scores >= 0, 1, -1)
    bias_scores = np.sum(bias_scores, axis=1) / bias_scores.shape[1]  # n_target_words, n_pairs
    return bias_scores
