import argparse
import os
from gensim.models import Word2Vec
from loguru import logger
from paths import corpus_path_dict, embed_folders


def train_w2v(filename, size=300, workers=48, window=5, min_count=5, num_iter=5):
    return Word2Vec(corpus_file=filename, size=size, workers=workers,
                    window=window, min_count=min_count, iter=num_iter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_dim", type=int, default=300)
    parser.add_argument("--num_threads", type=int, default=48)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--num_copies", type=int, default=32)
    parser.add_argument("--num_iter", type=int, default=5)
    parser.add_argument("--corpus", type=str, default='wikitext',
                        choices=['wikitext', 'reddit_ask_science', 'reddit_ask_historians'])
    parser.add_argument("--embed_folder", type=str, default=None)
    args = parser.parse_args()

    corpus_path = corpus_path_dict[args.corpus]
    embed_folder = os.path.join(
        embed_folders[args.corpus], 'sgns/') if args.embed_folder is None else args.embed_folder
    os.makedirs(embed_folder, exist_ok=True)

    for embed_no in range(1, args.num_copies + 1):
        embed_filename = f'{embed_no}.model'
        full_embed_filename = os.path.join(embed_folder, embed_filename)
        logger.info(f'embedding No.{embed_no} out of {args.num_copies}.')
        embed = train_w2v(corpus_path, size=args.num_dim, workers=args.num_threads,
                          window=args.window_size, min_count=args.min_count, num_iter=args.num_iter)
        embed.save(full_embed_filename)


if __name__ == '__main__':
    main()
