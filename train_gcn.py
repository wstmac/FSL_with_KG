import pickle
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer

from datasets import MiniImageNet
import models
from utils import get_splits, AverageMeter, setup_logger, get_classFile_to_wikiID
from graph import Graph


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_classifier(img_encoder, img_feature_dim, num_classes, data_loader, split, normalize, model_dir, device):
    img_encoder.eval()
    classifiers = torch.zeros(num_classes, img_feature_dim, dtype=torch.float32)
    # import ipdb; ipdb.set_trace()
    # return classifiers for each class
    with torch.no_grad():
        for _, (imgs, labels, _) in enumerate(tqdm(data_loader)):
            imgs = imgs.to(device)
            img_features, _, _ = img_encoder(imgs, norm=normalize)

            img_features = img_features.to('cpu')
            # # del img_features
            for i, label in enumerate(labels):
                classifiers[label] += img_features[i]
        
    classifiers /= 600
    
    with open(f'{model_dir}/{split}_classifiers.pkl', 'wb') as f:
        pickle.dump(classifiers, f, pickle.HIGHEST_PROTOCOL)

    return classifiers



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model_arch', default='conv4', choices=['conv4', 'resnet10', 'resnet18'], type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--num_epoch', default=90, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--model_saving_rate', default=30, type=int)
    parser.add_argument('--train', action='store_true')
    # parser.add_argument('--support_groups', default=10000, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluation_rate', default=10, type=int)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--img_encoder_path', type=str)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--save_settings', action='store_true')
    parser.add_argument('--layer', default=4, type=int)
    parser.add_argument('--classifiers_path', action='store_true')
    parser.add_argument('--optimizer', default='SGD', type=str)
    # parser.add_argument('--scheduler_milestones', nargs='+', type=int)


    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    model_arch = args.model_arch
    learning_rate = args.learning_rate
    start_epoch = args.start_epoch
    num_epoch = args.num_epoch
    model_saving_rate = args.model_saving_rate
    # toTrain = args.train
    # toEvaluate = args.evaluate
    evaluation_rate = args.evaluation_rate
    checkpoint = args.checkpoint
    # scheduler_milestones = args.scheduler_milestones
    save_settings = args.save_settings
    model_dir = f'./training_models/{args.model_dir}'
    img_encoder_path = f'{model_dir}/{args.img_encoder_path}'
    classifiers_path = args.classifiers_path
    normalize = args.normalize


    # ------------------------------- #
    # Config logger
    # ------------------------------- #
    train_logger = setup_logger('train_logger', f'{model_dir}/gcn_train.log')
    if save_settings:
        # ------------------------------- #
        # Saving training parameters
        # ------------------------------- #
        train_logger.info(f'{model_arch} Model: {img_encoder_path}')
        train_logger.info(f'Attention Layer: args.layer')
        train_logger.info(f'Learning rate: {learning_rate}')
        train_logger.info(f'Optimizer: {args.optimizer}')


    # ------------------------------- #
    # Load extracted knowledge graph
    # ------------------------------- #
    knowledge_graph = Graph()
    classFile_to_superclasses, superclassID_to_wikiID =\
        knowledge_graph.class_file_to_superclasses(1, [1,2])
    edges = knowledge_graph.edges
    nodes = knowledge_graph.nodes

    ####################
    # Prepare Data Set #
    ####################
    print('preparing dataset')
    base_cls, val_cls, support_cls = get_splits()
    base = MiniImageNet('base', base_cls, val_cls, support_cls, classFile_to_superclasses)
    base_loader = DataLoader(base, batch_size=256, shuffle=False, num_workers=4)

    # ------------------------------- #
    # Load image encoder model
    # ------------------------------- #
    # image encoder
    if model_arch == 'conv4':
        img_encoder = models.Conv4Attension(len(base_cls), len(superclassID_to_wikiID))

    if model_arch == 'resnet10':
        img_encoder = models.resnet10(len(base_cls), len(superclassID_to_wikiID))

    if model_arch == 'resnet18':
        img_encoder = models.resnet18(len(base_cls), len(superclassID_to_wikiID))

    img_encoder.load_state_dict(torch.load(f'{img_encoder_path}'))
    img_encoder.to(device)

    img_feature_dim = img_encoder.dim_feature



    # ------------------------------- #
    # get class classifiers
    # ------------------------------- #
    
    if classifiers_path:
        with open(f'{model_dir}/base_classifiers.pkl', 'rb') as f:
            classifiers = pickle.load(f)
    else:
        classifiers = get_classifier(img_encoder, img_feature_dim, len(base_cls), base_loader, 'base', normalize, model_dir, device)

    # import ipdb; ipdb.set_trace()

    # ------------------------------- #
    # Init GCN model
    # ------------------------------- #
    layer = 2
    layer_nums = [768, 2048, img_feature_dim]
    layer_nums_str = "".join([str(a)+' ' for a in layer_nums])
    if save_settings:
        train_logger.info(f'GCN layers: {layer_nums_str}')
    GCN = models.GCN(layer, layer_nums, edges)
    # GCN = models.GCN(edges)
    GCN.to(device)
    # import ipdb; ipdb.set_trace()
    # ------------------------------- #
    # Other neccessary parameters
    # ------------------------------- #
    classFile_to_wikiID = get_classFile_to_wikiID()
    base_cls_index = [nodes.index(classFile_to_wikiID[base.id_to_class_name[i]]) for i in range(len(base_cls))]
    # support_cls_index = [nodes.index(classFile_to_wikiID[base.id_to_class_name[i]]) for i in range(len(support_cls))]

    sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
    desc_embeddings = knowledge_graph.encode_desc(sentence_transformer)
    desc_embeddings =desc_embeddings.to(device)

    
    # ------------------------------- #
    # Training settings
    # ------------------------------- #
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.SGD(GCN.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # optimizer = torch.optim.Adam(GCN.parameters(), lr=learning_rate, weight_decay=1e-4)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss

    GCN.train()
    start = time.time()

    classifiers = classifiers.to(device)

    loss_target = torch.ones(classifiers.shape[0]).to(device)
    
    for epoch in range(start_epoch, start_epoch+num_epoch):
        base_embeddings = GCN(desc_embeddings)[base_cls_index]
        # import ipdb; ipdb.set_trace()
        # loss = criterion(base_embeddings, classifiers)
        loss = criterion(base_embeddings, classifiers, loss_target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        # print(loss.item())
        batch_time.update(time.time() - start)

        if epoch % 200 == 0:    # print every 30 epoch
            train_logger.info(f'[{epoch:3d}/{start_epoch+num_epoch-1}]'
                f' batch_time: {batch_time.avg:.2f} loss: {losses.avg:.3f}')  
            batch_time.reset() 
            losses.reset() 
            start = time.time() 
        if epoch % 1000 == 0:
            torch.save(GCN.state_dict(), f'{model_dir}/gcn_{epoch}.pth')



    train_logger.info("="*60)

# def evaluate(GCN, classifiers, device, support_cls_index, criterion, classifier_path=None):
#     GCN.eval()

#     if classifiers_path != None:
#         with open(f'{model_dir}/classifiers.pkl', 'rb') as f:
#             classifiers = pickle.load(f)
#     else:
#         classifiers = get_classifier(img_encoder, img_feature_dim, len(base_cls), base_loader, 'base', normalize, model_dir, device)




#     loss_target = torch.ones(classifiers.shape[0]).to(device)
#     with torch.no_grad():
#         support_embeddings = GCN(desc_embeddings)[support_cls_index]
#         loss = criterion(support_embeddings, classifiers, loss_target)
#     return loss.item()


if __name__ == '__main__':
    train()
