import sys
from typing import Text
sys.path.append('/home/wushuang/simple_shot/src')

import PIL.Image as Image
import requests
import tqdm
import pickle
import os
import random
import torch
import json
from wikidata.client import Client
from nltk.corpus import wordnet as wn
from io import BytesIO
import torchvision.models as models
from sentence_transformers import SentenceTransformer
from numpy import linalg as LA
from SPARQLWrapper import SPARQLWrapper, JSON

from datasets.transform import without_augment

global client, image_prop
global non_ambiguous, no_result, ambiguous, dataset
global search_result_dict, classFile_dict, lookup_table
global transform, sentence_transformer

sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')


PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE_DATA= '/home/wushuang/Dataset/miniImageNet/images'
dataset = 'ImageNet'

non_ambiguous = {}
no_result = {}
ambiguous = {}
search_result_dict = {}
classFile_dict = {}
with open('/home/wushuang/simple_shot/src/utils/lookup_table.json', 'r') as f:
            lookup_table = json.load(f)

client = Client()
image_prop = client.get('P18')

transform = without_augment()
API_ENDPOINT = "https://www.wikidata.org/w/api.php"


resnet18 = models.resnet18(pretrained=True)
encoder = torch.nn.Sequential(*list(resnet18.children())[:-1])


def get_classFile_dict():
    global classFile_dict
    for i in os.listdir(IMAGE_DATA):
        classFile = i[:9]
        if not classFile in classFile_dict.keys():
            classFile_dict[classFile] = [i]
        else:
            classFile_dict[classFile].append(i)


def get_image_by_file(classFile):
    images = torch.empty((60,3,84,84))
    for i, img_path in enumerate(random.sample(classFile_dict[classFile], 60)):
        img = Image.open(f'{IMAGE_DATA}/{img_path}').convert('RGB')
        img = transform(img)

        images[i] = img

    with torch.no_grad(): 
        encoded_images = encoder(images).view(60, -1)
        encoded_images = torch.mean(encoded_images, dim=0)
        
    return encoded_images


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_results(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def clean(str_):
    if str_[:31] == 'http://www.wikidata.org/entity/':
        return str_[31:]
    else:
        print('problem')
        return ''


def search_wikidata(query):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': query
    }
    r = requests.get(API_ENDPOINT, params = params)
    return r.json()['search']


def get_image_by_wikiID(wikiID):
    entity = client.get(wikiID, load=True)

    if len(entity.getlist(image_prop)) == 0:
        return None
    else:
        image_url = entity[image_prop].image_url

        if 'svg' in image_url:
            return None

        response = requests.get(image_url)
        image_bytes = BytesIO(response.content)
        img = Image.open(image_bytes).convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            encoded_image = encoder(img).view(1, -1)

        return torch.squeeze(encoded_image)


def get_def_by_wikiID(wikiID):
    global lookup_table
    query = """SELECT ?item ?itemLabel ?itemDescription WHERE {VALUES (?item) {(wd:""" + wikiID + """)}. \
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }} """
    result = get_results(query)['results']['bindings'][0]


    itemID = clean(result['item']['value'])
    itemContent = result['itemLabel']['value']

    if not 'itemDescription' in result:
        itemDescription = itemContent
    else:
        itemDescription = result['itemDescription']['value']
    lookup_table[itemID] = (itemContent, itemDescription)
    return itemDescription


def init_search():
    global non_ambiguous, no_result, ambiguous, search_result_dict
    wiki_search_results_path = f'{PATH}/{dataset}/{dataset}_wiki_search.pkl'
    if os.path.exists(wiki_search_results_path):
        search_result_dict = load_pickle(wiki_search_results_path)

    with open(f'{PATH}/{dataset}/ori_{dataset}.txt', 'r') as f:
        lines = f.readlines()
        for _, line in enumerate(tqdm.tqdm(lines)):
            class_file = line.split(':')[0].strip()
            raw_class_label = line.split(':')[1].strip()
            class_labels = []
            
            res_len = 0
            
            for label in raw_class_label.split(','):
                clean_label = label.strip()
                class_labels.append(clean_label)
                
                if not clean_label in search_result_dict.keys():
                    try:
                        result = search_wikidata(clean_label)
                        search_result_dict[clean_label] = result
                    except:
                        print(clean_label)
                else:
                    result = search_result_dict[clean_label]
                        
                if len(result) == 1:
                    non_ambiguous[class_file] = class_labels
                    res_len = 1
                    break
                else:
                    res_len = max(res_len, len(result))
            if res_len == 0:
                no_result[class_file] = class_labels
            elif res_len > 1:
                ambiguous[class_file] = class_labels
    save_pickle(wiki_search_results_path, search_result_dict)


def search_amnigous():
    file = open(f'{PATH}/{dataset}/{dataset}_with_wn.txt', 'w')

    for class_file, labels in tqdm.tqdm({**non_ambiguous, **ambiguous, **no_result}.items()):
        if class_file in classFile_dict.keys():
            set_list = []
            wn_results = []
            wn_labels = []
            for label in labels:
                # ----------------------------------------------------------- #
                # Search in wikidata
                # ----------------------------------------------------------- #
                label_ids = {res['id'] for res in search_result_dict[label]}
                set_list.append(label_ids)

                # ----------------------------------------------------------- #
                # Search in wordnet
                # ----------------------------------------------------------- #
                wn_label = label.strip().replace(' ','_')
                wn_labels.append(wn_label)
                wn_results.append(set(wn.synsets(wn_label, pos=wn.NOUN)))

            intersection = set.intersection(*set_list)
            union = set.union(*set_list)

            wn_intersection = set.intersection(*wn_results)

            save_label = ''
            for label in labels:
                save_label += label.replace(' ', '_') +'|'


        
            # ----------------------------------------------------------- #
            # Compare definition in wordnet and wikidata 
            # to select the wiki item
            # ----------------------------------------------------------- #
            flag = True # to determine length of wn_intersection, False->1

            if len(wn_intersection) == 1:
                pass
            else:
                flag = False

            wn_definition = next(iter(wn_intersection)).definition()
            enc_wn_def = sentence_transformer.encode(wn_definition)


        
            if len(intersection) == 0:
                distance = 1000
                selected_wikiID = ''

                for id in union:
                    if not id in lookup_table.keys():
                        definition = get_def_by_wikiID(id)
                    else:
                         definition = lookup_table[id][1]
                    enc_def = sentence_transformer.encode(definition)

                    dist = LA.norm(enc_wn_def-enc_def)
                    if dist < distance:
                        distance = dist
                        selected_wikiID = id

                file.write(f'{class_file}:{save_label} {selected_wikiID}\t{flag}\n')

            elif len(intersection) == 1:
                file.write(f'{class_file}:{save_label} {next(iter(intersection))}\t{flag}\n')
            else:
                distance = 1000
                selected_wikiID = ''

                for id in intersection:
                    if not id in lookup_table.keys():
                        definition = get_def_by_wikiID(id)
                    else:
                         definition = lookup_table[id][1]
                    enc_def = sentence_transformer.encode(definition)

                    dist = LA.norm(enc_wn_def-enc_def)
                    if dist < distance:
                        distance = dist
                        selected_wikiID = id

                file.write(f'{class_file}:{save_label} {selected_wikiID}\t{flag}\n')
        else:
            pass




        # ----------------------------------------------------------- #
        # Compare image in wikidata and imagenet 
        # to select the wiki item
        # ----------------------------------------------------------- #
        # if class_file in classFile_dict.keys():
        #     if len(intersection) == 0:
        #         distance = 1000
        #         selected_wikiID = ''
        #         base_image = get_image_by_file(class_file)

        #         for id in union:
        #             img = get_image_by_wikiID(id)
        #             if not img==None:
        #                 dist = torch.dist(base_image, img)
        #                 if dist < distance:
        #                     distance = dist
        #                     selected_wikiID = id

        #         file.write(f'{class_file}:{save_label} {selected_wikiID}')

        #     elif len(intersection) == 1:
        #         file.write(f'{class_file}:{save_label} {next(iter(intersection))}')
        #     else:
        #         distance = 1000
        #         selected_wikiID = ''
        #         base_image = get_image_by_file(class_file)

        #         for id in intersection:
        #             img = get_image_by_wikiID(id)
        #             if not img==None:
        #                 dist = torch.dist(base_image, img)
        #                 if dist < distance:
        #                     distance = dist
        #                     selected_wikiID = id

        #         file.write(f'{class_file}:{save_label} {selected_wikiID}')
        # else:
        #     pass

    file.close()
    with open('/home/wushuang/simple_shot/src/utils/lookup_table.json', mode='w') as lookup_table_file:
        json.dump(lookup_table, lookup_table_file)


get_classFile_dict()
init_search()
search_amnigous()


# ------------------------------- #
# Assist by wordnet
# ------------------------------- #
def search_by_wn():
    c = 0
    with open(f'{PATH}/{dataset}/ori_{dataset}.txt', 'r') as f:
        lines = f.readlines()
        for _, line in enumerate(lines):
            class_file = line.split(':')[0].strip()
            raw_class_label = line.split(':')[1].strip()
            clean_labels = []

            wn_results = []

            for label in raw_class_label.split(','):
                clean_label = label.strip().replace(' ','_')
                clean_labels.append(clean_label)
                wn_results.append(set(wn.synsets(clean_label, pos=wn.NOUN)))
                
            

            intersection = set.intersection(*wn_results)

            if len(intersection) != 1:
                print(f'{class_file}: {clean_labels} {len(intersection)}')
                c += 1
            

    print(f'There are {c} in total')

# search_by_wn()