import time
from numpy.core.shape_base import block
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import logging
from tqdm import tqdm

from datasets import MiniImageNet, SupportingSetSampler, prepare_nshot_task
from models import Conv4Classifier
from utils import get_splits, evaluation


def train():
    logging.basicConfig(filename='./logs/baseline.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    n = 1  # number of samples per supporting class 
    k = 5  # number of classes
    q = 15 # query image per class
    learning_rate = 0.1



    ####################
    # Prepare Data Set #
    ####################
    print('preparing dataset')
    base_cls, val_cls, support_cls = get_splits()

    base = MiniImageNet('base', base_cls, val_cls, support_cls)
    base_loader = DataLoader(base, batch_size=256, shuffle=True, num_workers=4)

    # val = MiniImageNet('val', base_cls, val_cls, support_cls)
    # val_loader = DataLoader(val, batch_size=256, shuffle=True, num_workers=4)

    # support = MiniImageNet('support', base_cls, val_cls, support_cls)
    # support_loader = DataLoader(support,
    #                         batch_sampler=SupportingSetSampler(support, n, k, q),
    #                         num_workers=4)

    #########
    # Model #
    #########
    model = Conv4Classifier(len(base_cls))
    model.to(device)


    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[int(.5*90),int(.75*90)], gamma=0.1)

    print('start to train')
    for epoch in range(90):
        running_loss = 0.0
        data_load_time = 0
        gpu_time = 0
        epoch_time = 0
        for i, data in enumerate(base_loader):
            time1 = time.time()
            inputs, labels = data[0].to(device), data[1].to(device)
            time2 = time.time()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time3=time.time()
            # print statistics
            running_loss += loss.item()
            data_load_time += time2-time1
            gpu_time += time3-time2
            epoch_time += time3-time1

            if i % 30 == 29:    # print every 2000 mini-batches
                logging.info('[%d, %3d] loss: %.3f ' %
                    (epoch + 1, i + 1, running_loss / 30))
                print('[%d, %5d] load_data:%2f gpu_time:%2f epoch_time:%.2f loss: %.3f ' %
                    (epoch + 1, i + 1, data_load_time, gpu_time, epoch_time, running_loss / 30))
                running_loss = 0.0
                data_load_time = 0
                gpu_time = 0
                epoch_time = 0

        scheduler.step()


    PATH = f'./baseline_{epoch}.pth'
    torch.save(model.state_dict(), PATH)


def evaluate():

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    # ------------------------------- #
    # Load Model
    # ------------------------------- #
    PATH = './baseline_89.pth'

    model = Conv4Classifier(64)
    model.load_state_dict(torch.load(PATH))

    # model = Conv4Classifier(64)        
    # checkpoint = torch.load('./model_best.pth.tar')
    # model_dict = model.state_dict()
    # params = checkpoint['state_dict']
    # params = {k: v for k, v in params.items() if k in model_dict}
    # model_dict.update(params)
    # model.load_state_dict(model_dict)


    model.to(device)
    model.eval()


    ####################
    # Prepare Data Set #
    ####################
    print('preparing dataset')
    n = 5  # number of samples per supporting class 
    k = 5  # number of classes
    q = 15 # query image per class
    episodes_per_epoch = 10000

    base_cls, val_cls, support_cls = get_splits()

    support = MiniImageNet('support', base_cls, val_cls, support_cls)
    support_loader = DataLoader(support,
                            batch_sampler=SupportingSetSampler(support, n, k, q, episodes_per_epoch),
                            num_workers=4)

    logging.basicConfig(filename=f'./logs/baseline_cosine_result_{k}-way_{n}-shot.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    print('start to evaluate')
    accs = 0
    for i, data in enumerate(tqdm(support_loader)):
        inputs, labels = prepare_nshot_task(n, k, q, data)
        embeddings = model(inputs, feature=True)

        acc = evaluation(embeddings, labels, n, k, q)
        logging.info(f'[{i:3d}]: {acc}%')
        accs += acc

    logging.info(f'Average ACC is {accs}/{len(support_loader)}={accs/len(support_loader)}')


if __name__ == '__main__':
    evaluate()
    


