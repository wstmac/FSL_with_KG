import json
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
import csv

from .miscellaneous import MINI_CLASSES_PATH, RELATIONSHIPS


def extract_subgraph(file_path):
    subjects = get_subjects()
    lookup_table = subjects.copy()
    # lookup_table['P31'] = 'instanceOf'
    lookup_table['P279'] = 'subclassOf'

    nodes, rels, lookup_table =\
            get_rel_linked_list(subjects, lookup_table)

    generateCSVFiles(nodes, rels, lookup_table, subjects, file_path)


def get_rel_linked_list(class_nodes, lookup_table):
    nodes = set()
    rels = set()

    processedNodes = {}
    for rel in RELATIONSHIPS:
        processedNodes[rel] = set()

    for i, wikiID in enumerate(class_nodes.keys()):
        print(f'[{i+1}/{len(class_nodes)}] Process {wikiID}: {lookup_table[wikiID]}')
        nodes.add(wikiID)
        for relID in RELATIONSHIPS.keys():
            nodes, rels, lookup_table, processedNodes[rel] = query(nodes, rels, lookup_table, wikiID, relID, processedNodes[rel])

    return nodes, rels, lookup_table


def query(nodes, rels, lookup_table, nodeID, relID, processedNodes):
    nodes, rels, lookup_table, result_nodes = _query(nodes, rels, lookup_table, nodeID, relID)
    processedNodes.add(nodeID)

    while result_nodes:
        newNodeID = result_nodes.pop()
        if not newNodeID in processedNodes:
            nodes, rels, lookup_table, sub_result_nodes = _query(nodes, rels, lookup_table, newNodeID, relID)
            processedNodes.add(newNodeID)
            result_nodes.update(sub_result_nodes)

    return nodes, rels, lookup_table, processedNodes


def _query(nodes, rels, lookup_table, nodeID, relID):
    query = """SELECT ?item ?itemLabel ?itemDescription WHERE {wd:""" + nodeID + """ wdt:""" + relID + """ ?item. \
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }} """
    results = get_results(query)

    result_nodes = set()

    for result in results['results']['bindings']:
        itemID = clean(result['item']['value'])
        itemContent = result['itemLabel']['value']

        if itemContent != itemID:
            nodes.add(itemID)
            rels.add((nodeID, relID, itemID))

            result_nodes.add(itemID)

            if itemID not in lookup_table.keys():
                if not 'itemDescription' in result:
                    itemDescription = itemContent
                else:
                    itemDescription = result['itemDescription']['value']
                lookup_table[itemID] = (itemContent, itemDescription)

    return nodes, rels, lookup_table, result_nodes


def check_query(nodes, rels):
    res_nodes = set()
    for relID in rels.keys():
        for nodeID in nodes.keys():
            res_nodes.add(nodeID)
            query = """SELECT ?item ?itemLabel WHERE {wd:""" + nodeID + """ wdt:""" + relID + """+ ?item. \
                SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }} """
            results = get_results(query)
            for result in results['results']['bindings']:
                res_nodes.add(clean(result['item']['value']))
    print(f'There are {len(res_nodes)} in total.')


def get_subjects():
    subjects = {}
    with open(MINI_CLASSES_PATH, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if len(line.split(' ')) == 2:
                label, wiki_id = line.split(' ')
                label = label.split(':')[-1].strip()
                wiki_id = wiki_id.strip()

                query= 'SELECT ?itemDescription\
                    WHERE {\
                        SERVICE wikibase:label {\
                        bd:serviceParam wikibase:language "en" .\
                        wd:'+ wiki_id +' schema:description ?itemDescription .\
                        }\
                    }'
                
                results = get_results(query)
                if 'itemDescription' in results['results']['bindings'][0]:
                    itemDescription = results['results']['bindings'][0]['itemDescription']['value']
                else:
                    itemDescription = label.replace('_', ' ')
                subjects[wiki_id] = (label, itemDescription)
                # subjects[wiki_id] = label
    return subjects


def generateCSVFiles(nodes, rels, lookup_table, classNodes, file_path):
    nodeCSV = open(f'{file_path}/node.csv', mode='w')
    nodeWriter = csv.writer(nodeCSV, delimiter=',')
    nodeWriter.writerow(['wikiID:ID','content',':LABEL'])

    relCSV = open(f'{file_path}/rel.csv', mode='w')
    relWriter = csv.writer(relCSV, delimiter=',')
    relWriter.writerow([':START_ID',':TYPE',':END_ID', 'wikiID'])

    for node in nodes:
        if node not in classNodes.keys():
            nodeLabel = 'Common_Node'
        else:
            nodeLabel = 'Class_Node'
        nodeWriter.writerow([node, lookup_table[node], nodeLabel])

    for rel in rels:
        relWriter.writerow([rel[0], lookup_table[rel[1]], rel[2], rel[1]])

    with open(f'{file_path}/lookup_table.json', mode='w') as lookup_table_file:
        json.dump(lookup_table, lookup_table_file)


def get_results(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def to_json(line):
    if line[-1] == ',':
        line = line[:-1]  # all lines should end with a ','

    # turn string to json
    if line[0] != '{' or line[-1] != '}':
        # then this line is not a proper json file we should deal with it later
        # raise ParsingException
        pass

    return json.loads(line)


def get_type(ent):
    return ent['type']


def get_id(ent):
    return ent['id']


def get_label(ent):
    """

    Parameters
    ----------
    ent: dict
        Dictionary coming from the parsing of a json line of the dump.

    Returns
    -------
    label: str
        Label of ent in english if available of any other language else.
    """

    labels = ent['labels']
    if len(labels) == 0:
        return 'No label {}'.format(ent['id'])
    if 'en' in labels.keys():
        return labels['en']['value']
    else:
        return labels[list(labels.keys())[0]]['value']


def relabel(x, labels):
    try:
        lab = labels[x]
        if ':' in lab:
            return lab[lab.index(':')+1:]
        else:
            return lab
    except KeyError:
        return x


def clean(str_):
    if str_[:31] == 'http://www.wikidata.org/entity/':
        return str_[31:]
    else:
        print('problem')
        return ''
