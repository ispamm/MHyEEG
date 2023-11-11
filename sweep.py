import argparse
from data.dataloader import MyDataLoader
from models.model import HyperFuseNet
from training import Trainer
import wandb
import numpy as np
import torch
import random
from multiprocessing import cpu_count

def tuning():
    
    # Set seed    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Set number of classes
    num_classes = 3

    wandb.init(project="MHyEEG")
    epochs =  wandb.config.epochs
    dropout_rate = wandb.config.dropout_rate
    lr = wandb.config.lr
    max_lr = 10 * lr
    min_mom = wandb.config.min_mom
    max_mom = wandb.config.max_mom
    
    train_loader, eval_loader, sample_weights = MyDataLoader(root=args.train_dir, label_kind=args.label_kind, batch_size=args.batch_size, num_workers=n_workers)
    net = HyperFuseNet(n=args.n, dropout_rate=dropout_rate)
    
    wandb.config.update({"max_lr": max_lr, 'sample_weights': sample_weights.tolist()})
    wandb.config.update(args) # to also log args
    wandb.watch(net)
    
    # Count NN parameters
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Number of parameters:', params)
    print()
    
    # Initialize optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay, eps=1e-7)
    
    # Train/Evaluate model
    trainer = Trainer(net, optimizer, epochs=epochs,
                      use_cuda=args.cuda, gpu_num=args.gpu_num,
                      checkpoint_folder=args.checkpoint_folder,
                      max_lr=max_lr, min_mom=min_mom,
                      max_mom=max_mom, l1_reg=args.l1_reg,
                      num_classes=num_classes,
                      sample_weights=sample_weights)
    
    trainer.train(train_loader, eval_loader)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', default=1, help="Number of workers, 'max' for maximum number")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--n', type=int, default=4, help="n parameter for PHC layers")
parser.add_argument('--l1_reg', type=bool, default=False)
parser.add_argument('--train_dir', type=str, default='./data/')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--checkpoint_folder', type=str, default='checkpoints')
parser.add_argument('--label_kind', type=str, default='Vlnc', help="Choose valence (Vlnc) or arousal (Arsl) label")
args = parser.parse_args()

seed = args.seed
n_workers = args.num_workers

if n_workers == 'max':
    n_workers = cpu_count()  # get the count of the number of CPUs in your system

sweep_configuration = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': 
    {
        'dropout_rate': {'min': 0.125, 'max': 0.4},
        'lr': {'min': 0.001, 'max': 0.008},
        'min_mom': {'min': 0.75, 'max': 0.89},  # min momentum in one cycle policy, in Adam case mom=beta 1
        'max_mom': {'min': 0.90, 'max': 0.99},  # max momentum in one cycle policy, in Adam case mom=beta 1
        'label': {'value': args.label_kind}
    }
}  

sweep_id = wandb.sweep(sweep=sweep_configuration, project='MHyEEG')
wandb.agent(sweep_id, function=tuning, count=3, project= "MHyEEG")
