import argparse
import os
import numpy as np
from data.dataloader import MyDataLoader
import torch
from training import Trainer
import wandb
import random
from multiprocessing import cpu_count
from models.model import HyperFuseNet

def main(args, n_workers):

    # Set number of classes
    num_classes = 3
    lr = args.max_lr / 10
    
    train_loader, eval_loader, sample_weights = MyDataLoader(train_file=args.train_file_path, 
                                                             test_file=args.test_file_path, 
                                                             batch_size=args.batch_size, 
                                                             num_workers=n_workers)
    
    net = HyperFuseNet(n=args.n, dropout_rate=args.dropout_rate)
    
    wandb.init(project="MHyEEG")
    wandb.config.update(args, allow_val_change=True)
    wandb.watch(net)
    
    # Count NN parameters
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Number of parameters:', params)
    print()
    
    # Initialize optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay, eps=1e-7)
    
    # Train/Evaluate model
    trainer = Trainer(net, optimizer, epochs=args.epochs,
                      use_cuda=args.cuda, gpu_num=args.gpu_num,
                      checkpoint_folder=args.checkpoint_folder,
                      max_lr=args.max_lr, min_mom=args.min_mom,
                      max_mom=args.max_mom, l1_reg=args.l1_reg,
                      num_classes=num_classes,
                      sample_weights=sample_weights)
    
    trainer.train(train_loader, eval_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_file_path', type=str, default='hci-tagging-database/torch_datasets/train_augmented_data_Arsl.pt', help='Path to training .pt file')
    parser.add_argument('--test_file_path', type=str, default='hci-tagging-database/torch_datasets/test_data_Arsl.pt', help='Path to test .pt file')
    parser.add_argument('--num_workers', default=1, help="Number of workers, 'max' for maximum number")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--n', type=int, default=4, help="n parameter for PHM layers")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout_rate', type=int, default=0.1789, help='0.1789 for arousal and 0.2118 for valence')
    parser.add_argument('--epochs', type=int, default=50, help="50 for arousal and 60 for valence")
    parser.add_argument('--max_lr', type=float, default=0.00000796, help="0.00000796 for arousal and 0.002489 for valence")
    parser.add_argument('--min_mom', type=float, default=0.7403, help="0.7403 for arousal and 0.8314 for valence")
    parser.add_argument('--max_mom', type=float, default=0.7985, help="0.7985 for arousal and 0.9735 for valence")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoints')
    parser.add_argument('--l1_reg', type=bool, default=False)
    args = parser.parse_args()

    seed = args.seed
    n_workers = args.num_workers

    if n_workers == 'max':
        n_workers = cpu_count()  # get the count of the number of CPUs in your system
    
    # Set seed    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    main(args, n_workers)