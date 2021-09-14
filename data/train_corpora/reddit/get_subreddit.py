import os
import bz2
import json
import re

import tqdm
import spacy

months = [
    '01', '02', '03', '04',
    '05', '06', '07', '08',
    '09', '10', '11', '12'
]
nlp = spacy.load('en_core_web_sm')


def preprocess(content):
    def remove_ss(c):
        c = re.sub(r"\r", "", c)
        c = re.sub(r"\n", "", c)
        c = re.sub(r"\t", " ", c)
        return c

    def remove_url(c):
        return re.sub(r"http\S+", "", c)

    def tokenize(c):
        return [' '.join([token.lower_ for token in sent]) for sent in nlp(c).sents]

    return tokenize(remove_url(remove_ss(content)))


def main():
    os.makedirs('subs/', exist_ok=True)
    sub_names = ['askscience', 'askhistorians']
    sub_io_streams = {sub_name: open(os.path.join(
        'subs/', f'{sub_name}.txt'), 'a') for sub_name in sub_names}

    for idx, month in enumerate(months):
        reddit_bz2_file = f'RC_2014-{month}.bz2'
        print(f'processing file #{idx}: {reddit_bz2_file}')

        try:
            with bz2.open(reddit_bz2_file) as fin:
                for line in tqdm.tqdm(fin):
                    line = json.loads(line.decode('utf-8').strip())
                    sub_name, content = line['subreddit'].lower(), line['body']
                    if content == '[deleted]':
                        continue
                    if sub_name in sub_io_streams:
                        sentences = preprocess(content)
                        for sent in sentences:
                            print(sent, file=sub_io_streams[sub_name])
        except FileNotFoundError:
            print(f'{reddit_bz2_file} is not found.')
            continue

    for sub_name in sub_io_streams:
        sub_io_streams[sub_name].close()


if __name__ == '__main__':
    main()
