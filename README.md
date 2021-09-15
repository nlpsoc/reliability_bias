# Assessing the Reliability of Word Embedding Gender Bias Measures


**Prepare Folders**
```shell
mkdir data/embed/
mkdir data/results/
```

**Requirements**

Our experiments are performed on ```Python 3.7```. 
```shell
conda create -n reliability_bias python=3.7
conda activate reliability_bias
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Download and Preprocess Data**

Follow the instructions in ```data/train_corpora/```.

**Train Word Embeddings** 

1. Skip-gram with Negative Sampling  
We use 48 threads as default. You can change it to fit your own machine.
   
```shell
python train_sgns.py --corpus wikitext --num_threads 48
python train_sgns.py --corpus reddit_ask_science --num_threads 48
python train_sgns.py --corpus reddit_ask_historians --num_threads 48
```

2. GloVe
First clone the repository from GitHub and compile
```shell
git clone https://github.com/stanfordnlp/glove
cd glove && make
```
Then make your own script to train with different corpora, 
and save the embeddings of WikiText-103, r/AskScience, and r/AskHistorians 
at ```EMBEDDING_FOLDER/glove```. 
For ```EMBEDDING_FOLDER``` see ```embed_folders``` in ```paths.py```. 

**Calculate Word Embedding Gender Bias Scores**

After training word embeddings, 
we calculate gender bias scores of words regarding each word embedding model.
```shell
%SGNS
python calc_bias_scores.py --embed_folder data/embed/wikitext-103 --vocab_path data/embed/wikitext-103/vocab.txt --embed_type sgns --bias_score_path data/results/bias_scores/wikitext-103/sgns.pkl
python calc_bias_scores.py --embed_folder data/embed/wikitext-103 --vocab_path data/embed/wikitext-103/vocab.txt --embed_type glove --bias_score_path data/results/bias_scores/wikitext-103/glove.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askscience --vocab_path data/embed/reddit/askscience/vocab.txt --embed_type sgns --bias_score_path data/results/bias_scores/reddit/askscience/sgns.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askscience --vocab_path data/embed/reddit/askscience/vocab.txt --embed_type glove --bias_score_path data/results/bias_scores/reddit/askscience/sgns.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askhistorians --vocab_path data/embed/reddit/askhistorians/vocab.txt --embed_type sgns --bias_score_path data/results/bias_scores/reddit/askhistorians/sgns.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askhistorians --vocab_path data/embed/reddit/askhistorians/vocab.txt --embed_type glove --bias_score_path data/results/bias_scores/reddit/askhistorians/glove.pkl
```

**Estimate Reliability and Run Experiments** 

Run ```reliability_analyses.ipynb``` 
after you have calculated word embedding gender bias scores. 

If you want to train your own word embeddings and run reliability estimation and analyses, 
please refer to ```reliability_metrics.py```. 
```ReliabilityEstimator``` can help you get the job done. 

**Regression Analyses**  

See ```mlr/```. 


**Citation**

If you find this repository useful, please consider citing our paper
```
@inproceedings{du-etal-2021-assessing,
    title = "Assessing the Reliability of Word Embedding Gender Bias Measures",
    author = "Du, Yupei  and
      Fang, Qixiang  and
      Nguyen, Dong",
    booktitle = "EMNLP",
    year = "2021"
}
```

**Contact** 

If you have questions/issues, 
either open an issue or contact Yupei Du (y.du@uu.nl) directly. 