import os
import torch

# ------------------------------- #
# Constant Variables
# ------------------------------- #
PATH = os.path.dirname(os.path.realpath(__file__))

NODE_PATH = f'{PATH}/node.csv'
REL_PATH = f'{PATH}/rel.csv'
MINI_CLASSES_PATH = f'{PATH}/miniDatasets.txt'
LOOKUP_TABLE_PATH = f'{PATH}/lookup_table.json'

RELATIONSHIPS = {'P279':'subclassOf'}
EPSILON = 1e-8

# ------------------------------- #
# Other Functions
# ------------------------------- #
def get_wikiID_to_classFile(file_path=MINI_CLASSES_PATH):
    """wikiID to classFile (e.g: 'Q140': 'n02129165')"""

    wikiID_to_classFile = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split(' ')) == 2:
                label, wiki_id = line.split(' ')
                class_file = label.split(':')[0].strip()
                wiki_id = wiki_id.strip()
                wikiID_to_classFile[wiki_id] = class_file
    return wikiID_to_classFile


def get_classFile_to_wikiID(file_path=MINI_CLASSES_PATH):
    """classFile to wikiID (e.g: 'n02129165': 'Q140')"""

    classFile_to_wikiID = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split(' ')) == 2:
                label, wiki_id = line.split(' ')
                class_file = label.split(':')[0].strip()
                wiki_id = wiki_id.strip()
                classFile_to_wikiID[class_file] = wiki_id
    return classFile_to_wikiID



def extract_embedding_by_labels(nodes, kg_embeddings, labels, class_name_to_id, classFile_to_wikiID):
    """Extract corresponding kg embeddings w.r.t given image labels"""

    # convert class_name_to_id to id_to_class_name
    id_to_class_name = {}
    for item in class_name_to_id.items():
        id_to_class_name[item[1]] = item[0]

    # find file name by labels
    file_names = [id_to_class_name[label.item()] for label in labels]

    # find wikiID by file names
    wikiID_list = [classFile_to_wikiID[file_name] for file_name in file_names]

    # find wikiID indexs in nodes
    wikiID_indexs = [nodes.index(wikiID) for wikiID in wikiID_list]
    # import ipdb; ipdb.set_trace()
    # get corresponding embeddings
    return torch.FloatTensor(kg_embeddings[wikiID_indexs, :])


# ----------------------------------------#
# Get corresbonding node indexs by labels
# --------------------------------------- #
def find_nodeIndex_by_imgLabels(nodes, labels, id_to_class_name, classFile_to_wikiID):
    """Extract corresponding wikiIDs w.r.t given image labels"""

    # find file name by labels
    file_names = [id_to_class_name[label.item()] for label in labels]

    # find wikiID by file names
    wikiID_list = [classFile_to_wikiID[file_name] for file_name in file_names]

    # find wikiID indexs in nodes
    wikiID_indexs = [nodes.index(wikiID) for wikiID in wikiID_list]
    # import ipdb; ipdb.set_trace()
    # get corresponding embeddings
    return wikiID_indexs