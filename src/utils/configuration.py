import configargparse


def parser_args():
    parser = configargparse.ArgParser(description='PyTorch ImageNet Training')
    parser.add('-c', '--config', required=True,
               is_config_file=True, help='config file')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to be used.')
    ### dataset
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--num-classes', type=int, default=64,
                        help='use all data to train the network')
    parser.add_argument('--num-spclasses', type=int, default=-1,
                        help='number of super classes in the knowledge graph')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--disable-train-augment', action='store_true',
                        help='disable training augmentation')
    parser.add_argument('--disable-random-resize', action='store_true',
                        help='disable random resizing')
    parser.add_argument('--enlarge', action='store_true',
                        help='enlarge the image size then center crop')
    ### network setting
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='network architecture')
    parser.add_argument('--scheduler', default='step', choices=('step', 'multi_step', 'cosine'),
                        help='scheduler, the detail is shown in train.py')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-stepsize', default=30, type=int,
                        help='learning rate decay step size ("step" scheduler) (default: 30)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'Adam'))
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov for SGD, disable it in default')
    parser.add_argument('--loss-alpha', default=0, type=float,
                        help='alpha for loss weight of CrossEntropyLoss')
    parser.add_argument('--loss-beta', default=0, type=float,
                        help='beta for loss weight of BCELoss')
    parser.add_argument('--loss-gamma', default=0, type=float,
                        help='gamma for loss weight of KG-Visual loss')
    parser.add_argument('--pool-type', default='avg_pool', choices=['avg_pool', 'max_pool'], type=str,
                        help='pool type for the channel-wise attention mechanism')
    parser.add_argument('--top-k', default=16, type=int,
                        help='select top-k feature maps of super feature')
    ### meta val setting
    parser.add_argument('--meta-test-iter', type=int, default=10000,
                        help='number of iterations for meta test')
    parser.add_argument('--meta-val-iter', type=int, default=500,
                        help='number of iterations for meta val')
    parser.add_argument('--meta-val-way', type=int, default=5,
                        help='number of ways for meta val/test')
    parser.add_argument('--meta-val-shot', type=int, default=1,
                        help='number of shots for meta val/test')
    parser.add_argument('--meta-val-query', type=int, default=15,
                        help='number of queries for meta val/test')
    parser.add_argument('--meta-val-interval', type=int, default=10,
                        help='do mate val in every D epochs')
    parser.add_argument('--meta-val-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='meta-val/test evaluate metric')
    parser.add_argument('--num_NN', type=int, default=1,
                        help='number of nearest neighbors, set this number >1 when do kNN')
    parser.add_argument('--eval_fc', action='store_true',
                        help='do evaluate with final fc layer.')
    ### meta train setting
    parser.add_argument('--do-meta-train', action='store_true',
                        help='do prototypical training')
    parser.add_argument('--meta-train-iter', type=int, default=100,
                        help='number of iterations for meta val')
    parser.add_argument('--meta-train-way', type=int, default=30,
                        help='number of ways for meta val')
    parser.add_argument('--meta-train-shot', type=int, default=1,
                        help='number of shots for meta val')
    parser.add_argument('--meta-train-query', type=int, default=15,
                        help='number of queries for meta val')
    parser.add_argument('--meta-train-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='meta-train evaluate metric')
    ### others
    parser.add_argument('--split-dir', default=None, type=str,
                        help='path to the folder stored split files.')
    parser.add_argument('--save-path', default='result/default', type=str,
                        help='path to folder stored the log and checkpoint')
    parser.add_argument('--model-dir', default=None, type=str,
                        help='path to folder stored the model, log and checkpoint')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-tqdm', action='store_true',
                        help='disable tqdm.')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--cutmix_prob', default=0, type=float,
                        help='cutmix probability. Open cutmix augmentation when cutmix_prob>0 and beta >0')
    parser.add_argument('--beta', default=-1., type=float,
                        help='cutmix beta. Open cutmix augmentation when cutmix_prob>0 and beta >0')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the final result')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='path to the pretrained model, used for fine-tuning')
    parser.add_argument('--log-info', action='store_true',
                        help='Write trianing parameters into the log file')
    parser.add_argument('--debug', action='store_true',
                        help='Write trianing parameters into the log file')

    ### MoCo
    parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')


    parser.add_argument('--model-name', default='checkpoint_399.pth.tar', type=str,
                        help='checkpoint name of model')
    return parser.parse_args()
