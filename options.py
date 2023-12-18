import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=1,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients')
    parser.add_argument('--batch_size', type=int, default=4096*2,
                        help="batch size")
    parser.add_argument('--num_batches_per_step', type=int, default=1,
                        help="the number of local batches")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--warmup_lr_epochs', type=int, default=5,
                        help="warmup epochs for lr")
    parser.add_argument('--attack_method', type=int, default=0,
                        help="0 normal 1 FoE")
    parser.add_argument('--foe_rate', type=int, default=3,
                        help='FoE attack rate')
    parser.add_argument('--meank_rate', type=int, default=50,
                        help='meank_rate defense')
    parser.add_argument('--en_defence', type=int, default=0,
                        help='enable defense')
    parser.add_argument('--group_size', type=int, default=3,
                        help=' group_size')
    parser.add_argument('--attack_rate', type=float, default=1,
                        help=' attack_rate')
    parser.add_argument('--final_aggregation_ratio', type=float, default=1,
                        help=' final_aggregation_ratio')
    parser.add_argument('--en_partial_att', type=int, default=100,
                        help=' en_partial_att')
    parser.add_argument('--attack_scale', type=float, default=1,
                        help=' attack_scale')
    

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, type=int, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        args.gpu = 0
    
    if args.gpu is not None:
        args.gpu_id = args.gpu

    args.schedule_lr_per_epoch = True

    return args
