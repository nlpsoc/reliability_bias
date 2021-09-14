**Download Data**

```shell
wget -c https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
```

**Decompress and remove unnecessary files**

```shell
unzip wikitext-103-v1.zip
mv wikitext-103/wiki.train.tokens wikitext-103-train.txt
rm -r wikitext-103/ wikitext-103-v1.zip
```

**Create Training Corpus for GloVe by substitute <unk> with <raw_unk>** 
```shell
cat wikitext-103-train.txt | sed -e 's/<unk>/<raw_unk>/g' > wikitext-103-train.glove.txt
```