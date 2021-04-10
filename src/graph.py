import numpy as np
from numpy.linalg import matrix_power
import json
import torch

from utils.miscellaneous import NODE_PATH, REL_PATH, LOOKUP_TABLE_PATH, RELATIONSHIPS, get_wikiID_to_classFile
# from utils.extract_subgraph import get_subjects


class Graph():
    def __init__(self):
        with open(LOOKUP_TABLE_PATH, 'r') as f:
            self.lookup_table = json.load(f)
        self.rel_type = self.get_rel_type()
        self.nodes, self.class_nodes = self.get_nodes()
        self.class_nodes_index = self.get_class_nodes_index()
        self.num_nodes = len(self.nodes)
        self.edges, self.num_edges = self.get_edges()

    
    def get_rel_type(self):
        rel_type = {}
        for i, relID in enumerate(RELATIONSHIPS.keys()):
            rel_type[relID] = i+1
        return rel_type


    def get_nodes(self):
        nodes = []
        classNodes = []
        with open(NODE_PATH, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                wikiID, _ = line.split(',', 1)
                nodes.append(wikiID)
                if 'Class_Node' in line:
                    classNodes.append(wikiID)
        return nodes, classNodes

    def get_class_nodes_index(self):
        class_node_index = []
        for class_node in self.class_nodes:
            class_node_index.append(self.nodes.index(class_node))
        return class_node_index


    def get_edges(self):
        edges = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        num_edges = 0
        with open(REL_PATH, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                num_edges += 1
                startID, _, endID, typeID = line.split(',')
                ind_startID = self.nodes.index(startID.strip())
                ind_endID = self.nodes.index(endID.strip())
                edges[ind_startID, ind_endID] = self.rel_type[typeID.strip()]
        return edges, num_edges


    def get_node_info_by_index(self, index):
        wikiID = self.nodes[index]
        wikiContent = self.lookup_table[wikiID]
        return wikiID, wikiContent


    def get_node_index_by_wikiID(self, wikiID):
        return self.nodes.index(str(wikiID)), self.lookup_table[wikiID]


    def get_node_index_by_wikiContent(self, wikiContent):
        for key, value in self.lookup_table.items():
            if wikiContent == value:
                node_index, _ = self.get_node_index_by_wikiID(key)
                return node_index, key
        return None


    def get_super_classes(self, nHops, rels):
        """Get super classes that is nHops away from the class nodes with relation type restriction rel.

        # Arguments
            nHops: super class nodes that is nHops away from class nodes 
            rel: edge type restrction

        # Returns
            results: dict where key is class and values are super classes
        """
        result = {i:set() for i in self.class_nodes_index}

        adj_matrix = self.edges.copy()
        adj_matrix[np.isin(adj_matrix, rels, invert=True)] = 0

        if nHops == 1:
            for i in self.class_nodes_index:
                result[i] = set(np.where(adj_matrix[i]!=0)[0])
        else:
            dist_matrix_pre = matrix_power(adj_matrix, nHops - 1)
            dist_matrix = matrix_power(adj_matrix, nHops)
            for i in self.class_nodes_index:
                result[i] = set(np.where(dist_matrix[i]!=0)[0]) - set(np.where(dist_matrix_pre[i]!=0)[0])

        return result


    def class_file_to_superclasses(self, nHops, rels):
        """Generate super class for each class in dataset.
        For each class, it will have multiple super-classes.
        So the task should be multi-label classification task.

        # Arguments
            nHops: super class nodes that is nHops away from class nodes 
            rel: edge type restrction

        # Returns
            classFile_to_superclasses: {class file name in dataset: [index]}
            superclassID_to_wikiID: {super classes label int: (spWikiID, spWikiContent)}
        """
        wikiID_to_classFile = get_wikiID_to_classFile()
        wikiID_to_superclasses = self.get_super_classes(nHops, rels)

        superclass_set = set()
        for _, value in wikiID_to_superclasses.items():
            superclass_set.update(value)
        superclass_sorted_list = list(sorted(superclass_set))

        classFile_to_superclasses = {}
        for key, value in wikiID_to_superclasses.items():
            wikiID, _ = self.get_node_info_by_index(key)
            classFile = wikiID_to_classFile[wikiID]

            superclass_indexs = []
            for v in value:
                superclass_indexs.append(superclass_sorted_list.index(v))

            #convert to multi-hot 
            sp_multi_hot = np.zeros(len(superclass_sorted_list), dtype=np.float32)
            sp_multi_hot[superclass_indexs] = 1

            classFile_to_superclasses[classFile] = sp_multi_hot


        superclassID_to_wikiID = {}
        for i, sp_index in enumerate(superclass_sorted_list):
            superclassID_to_wikiID[i] = self.get_node_info_by_index(sp_index)

        return classFile_to_superclasses, superclassID_to_wikiID


    # ------------------------------- #
    # GCN methods
    # ------------------------------- #
    def encode_desc(self, model):
        descriptions = []

        for node in self.nodes:
            desc = self.lookup_table[node][1]
            descriptions.append(desc)

        desc_embeddings = model.encode(descriptions)
        return torch.Tensor(desc_embeddings)

