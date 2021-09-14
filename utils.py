import json
import pickle
import numpy as np


def write_json(data, filename, formatted=True):
    with open(filename, 'w') as fout:
        if formatted:
            json.dump(data, fout, indent=4, separators=(',', ':'))
        else:
            json.dump(data, fout)
    return filename


def load_json(filename):
    with open(filename) as fin:
        data = json.load(fin)
    return data


def write_pickle(data, filename):
    with open(filename, 'wb') as fout:
        pickle.dump(data, fout)
    return filename


def load_pickle(filename):
    with open(filename, 'rb') as fin:
        data = pickle.load(fin)
    return data


def load_txt_list(filename):
    with open(filename) as fin:
        data = [line.strip() for line in fin]
    return data


def normalize_embed(embed):
    return embed / np.linalg.norm(embed)


def dict_to_list(d):
    return [d[k] for k in d]
