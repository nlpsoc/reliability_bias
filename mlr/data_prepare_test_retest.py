import pickle as pk
import pandas as pd
import statsmodels.api as sm

# %% paths to the pkl files storing the reliability scores
path_ls = ['../data/results/reliability/reddit/askhistorians/glove.pkl',
           '../data/results/reliability/reddit/askhistorians/sgns.pkl',
           '../data/results/reliability/reddit/askscience/glove.pkl',
           '../data/results/reliability/reddit/askscience/sgns.pkl',
           '../data/results/reliability/wikitext-103/glove.pkl',
           '../data/results/reliability/wikitext-103/sgns.pkl']

# %% make a data frame to store all the test-retest reliability scores
test_retest_df = pd.DataFrame()

for path in path_ls:
    with open(path, 'rb') as f:
        data = pk.load(f)

    data_wide = pd.DataFrame(data['test-retest'][0])
    data_wide = data_wide.rename_axis('target_w').reset_index()
    data_long = pd.melt(data_wide,
                        id_vars=['target_w'],
                        value_vars=['db_wa', 'ripa', 'nbm'],
                        var_name='metric',
                        value_name='score')

    if 'glove' in path:
        data_long['model'] = 'glove'
    else:
        data_long['model'] = 'sgns'

    if 'askhistorians' in path:
        data_long['corpus'] = 'askhistorians'

    elif 'askscience' in path:
        data_long['corpus'] = 'askscience'

    else:
        data_long['corpus'] = 'wikitext'

    test_retest_df = pd.concat([test_retest_df, data_long])


# %% paths to the csv files storing the word property data
path_ls = ['../data/results/property/reddit_ask_historians.csv',
           '../data/results/property/reddit_ask_science.csv',
           '../data/results/property/wikitext.csv']

# %% create a data frame to store the word property data across the data sets
feature_df = pd.DataFrame()

for path in path_ls:
    data = pd.read_csv(path).rename({'Unnamed: 0': 'target_w'}, axis=1)

    if 'historians' in path:
        data['corpus'] = 'askhistorians'

    elif 'science' in path:
        data['corpus'] = 'askscience'

    else:
        data['corpus'] = 'wikitext'

    feature_df = pd.concat([feature_df, data])


# %% left join the word property data with the reliability data frame
full_df = pd.merge(test_retest_df, feature_df, how="left", on=["target_w", "corpus"])

# %% check out the rows with NAs
full_df[full_df.isna().any(axis=1)]

# %% there are 1200582 rows in total with missing values
full_df[full_df.isna().any(axis=1)].shape[0]

# %% that's 0.721213 missing
full_df[full_df.isna().any(axis=1)].shape[0] / full_df.shape[0]

# %% there are 156651 unique target words without features
len(full_df[full_df.isna().any(axis=1)].target_w.unique())

# %% there are 187886 unique target words (with reliability scores) in total
len(full_df.target_w.unique())

# %% there are 31235 unique target words (with both reliability and features) in total
len(full_df.target_w.unique()) - len(full_df[full_df.isna().any(axis=1)].target_w.unique())

# %% there are 31236 unique target words (with features) in total
len(feature_df.target_w.unique())

# %% check how many missing columns there are per row
full_df.isna().sum(axis=1).unique()
# this tells us that a target word either has all or no features

# %% keep only the rows without missing values
complete_df = full_df.dropna()

# %% save
complete_df.to_csv('complete_test_retest.csv', index=False) #save under the same directory (reliability_bias/mlr)