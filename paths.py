import os

# data paths
data_dir = 'data/'

# training corpora
train_corpora_dir = os.path.join(data_dir, 'train_corpora/')
wikitext_corpus_path = os.path.join(train_corpora_dir, 'wikitext-103/wikitext-103-train.txt')
reddit_corpora_dir = os.path.join(train_corpora_dir, 'reddit/')
reddit_ask_science_path = os.path.join(reddit_corpora_dir, 'subs/askscience.txt')
reddit_ask_historian_path = os.path.join(reddit_corpora_dir, 'subs/askhistorians.txt')
corpus_path_dict = {
    'wikitext': wikitext_corpus_path,
    'reddit_ask_science': reddit_ask_science_path,
    'reddit_ask_historians': reddit_ask_historian_path
}


# embedding paths
embed_dir = os.path.join(data_dir, 'embed/')
embed_folders = {
    'wikitext': os.path.join(embed_dir, 'wikitext-103/'),
    'reddit_ask_science': os.path.join(embed_dir, 'reddit/askscience/'),
    'reddit_ask_historians': os.path.join(embed_dir, 'reddit/askhistorians/')
}

# lexicon paths
word_list_dir = os.path.join(data_dir, 'word_lists/')
nips_16_gbp_path = os.path.join(word_list_dir, 'gender_base_pairs.json')
pnas_18_gbp_path = os.path.join(word_list_dir, 'rnd_gender_base_pairs.json')
weat_lexicon_path = os.path.join(word_list_dir, 'weat_words.py')
nips_16_profession_list_path = os.path.join(word_list_dir, 'professions.json')
pnas_18_occupation_list_path = os.path.join(word_list_dir, 'occupations1950.txt')
pnas_18_adjective_list_path = os.path.join(word_list_dir, 'adjectives_williamsbest.txt')
google_10k_most_frequent_list_path = os.path.join(word_list_dir, 'google-10000-english.txt')

# results paths
results_dir = 'data/results/'

# bias scores
bias_scores_dir = os.path.join(results_dir, 'bias_scores/')
bias_scores_folders = {
    'wikitext': os.path.join(bias_scores_dir, 'wikitext-103/'),
    'reddit_ask_science': os.path.join(bias_scores_dir, 'reddit/askscience/'),
    'reddit_ask_historians': os.path.join(bias_scores_dir, 'reddit/askhistorians/')
}

# word embedding stability
embed_stability_dir = os.path.join(results_dir, 'embed_stability/')

reliability_dir = os.path.join(results_dir, 'reliability/')
reliability_dict_dirs = {
    'wikitext': os.path.join(reliability_dir, 'wikitext-103'),
    'reddit_ask_science': os.path.join(reliability_dir, 'reddit/askscience'),
    'reddit_ask_historians': os.path.join(reliability_dir, 'reddit/askhistorians')
}

# properties
property_dir = os.path.join(results_dir, 'property/')

# images
images_dir = os.path.join(results_dir, 'images/')
reliability_images_dir = os.path.join(images_dir, 'reliability/')

test_retest_image_dir = os.path.join(reliability_images_dir, 'test_retest/')
gbps_test_retest_image_dir = os.path.join(test_retest_image_dir, 'gbp/')
target_words_test_retest_image_dir = os.path.join(test_retest_image_dir, 'target_word/')

inter_rater_image_dir = os.path.join(reliability_images_dir, 'inter_rater/')
gbps_inter_rater_image_dir = os.path.join(inter_rater_image_dir, 'gbp/')
target_words_inter_rater_image_dir = os.path.join(inter_rater_image_dir, 'target_word/')

internal_image_dir = os.path.join(reliability_images_dir, 'internal/')
