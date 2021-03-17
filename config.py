import os

PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = '/home/wushuang/Desktop/Dataset'
NODE_PATH = './extracted_graph/node.csv'
REL_PATH = './extracted_graph/rel.csv'
MINI_CLASSES_PATH = './data/miniDatasets.txt'
LOOKUP_TABLE_PATH = './extracted_graph/lookup_table.json'

RELATIONSHIPS = {'P31':'instanceOf', 'P279':'subclassOf'}
EPSILON = 1e-8


if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')\

