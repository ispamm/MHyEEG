from tqdm import tqdm
from earlystopping import EarlyStopping
import torch
import time
import torch.nn as nn
import torch.optim.lr_scheduler as sched
import wandb
from sklearn.metrics import f1_score   

class Trainer():
    def __init__(self, net, optimizer, epochs,
                      use_cuda=True, gpu_num=0,
                      checkpoint_folder="./checkpoints",
                      max_lr=0.1, min_mom=0.7,
                      max_mom=0.99, l1_reg=False,
                      num_classes=3,
                      sample_weights=None,
                      es_mode='max',
                      patience=10):

        self.optimizer = optimizer
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.checkpoints_folder = checkpoint_folder
        # self.max_lr = max_lr
        self.min_mom = min_mom,
        self.max_mom = max_mom
        self.l1_reg = l1_reg
        self.num_classes = num_classes
        self.es_mode = es_mode
        self.patience = patience

        sample_weights = torch.tensor(sample_weights, dtype=torch.float32) if len(sample_weights)>0 else None
        self.criterion = nn.CrossEntropyLoss(weight=sample_weights)
        self.val_criterion = nn.CrossEntropyLoss()
        
        if self.use_cuda:
            if sample_weights is not None:
                self.criterion.weight = sample_weights.clone().detach().cuda('cuda:%i' %self.gpu_num)

            print(f"Running on GPU?", self.use_cuda, "- gpu_num: ", self.gpu_num)
            self.net = net.cuda('cuda:%i' %self.gpu_num)
            
        else:
            self.net = net

    def train(self, train_loader, eval_loader, **sched_kwargs):
        
        # name for checkpoint
        run_name = wandb.run.name

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, path=self.checkpoints_folder + "/best_" + run_name + ".pt", mode=self.es_mode)

        scheduler = sched.OneCycleLR(self.optimizer, epochs=self.epochs, steps_per_epoch=len(train_loader), 
                                         anneal_strategy='linear', cycle_momentum=True, base_momentum=self.min_mom, max_momentum=self.max_mom, 
                                         three_phase=True, **sched_kwargs)
        # scheduler = sched.StepLR(self.optimizer, step_size=5, gamma=0.1)
        
        best_f1 = 0
        best_loss = 0
        best_acc = 0
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            start = time.time()
            running_loss_train = 0.0
            running_loss_eval = 0.0
            train_total = 0.0
            train_correct = 0.0
            train_y_pred = torch.empty(0)
            train_y_true = torch.empty(0)
            total = 0.0
            correct = 0.0
            y_pred = torch.empty(0)
            y_true = torch.empty(0)
            
            self.net.train()  # switch net to training setting 
           
            for inputs, labels in tqdm(train_loader, total=len(train_loader), desc='Train round', unit='batch', leave=False):  # for each batch
                eye, gsr, eeg, ecg = inputs  # Tensors

                if self.use_cuda:
                    eye, gsr, eeg, ecg = eye.cuda('cuda:%i' %self.gpu_num), gsr.cuda('cuda:%i' %self.gpu_num), eeg.cuda('cuda:%i' %self.gpu_num), ecg.cuda('cuda:%i' %self.gpu_num), 
                    labels = labels.cuda('cuda:%i' %self.gpu_num)
                
                self.optimizer.zero_grad()  # clears grad for every parameter x in the optimizer, to not accumulate the gradients from multiple passes

                outputs = self.net(eye, gsr, eeg, ecg)
                loss = self.criterion(outputs, labels)

                if self.l1_reg:
                    print("Adding L1 regularization to A")
                    # Add L1 regularization to A
                    regularization_loss = 0.0
                    for child in self.net.children():
                        for layer in child.modules():
                            if isinstance(layer, PHConv):
                                for param in layer.a:
                                    regularization_loss += torch.sum(abs(param))
                    loss += 0.001 * regularization_loss


                loss.backward()  # computes dloss/dx, for every parameter x which has requires_grad=True, and save it into x.grad
                self.optimizer.step()  # updates the value of x using the computed x.grad value
                scheduler.step()
                wandb.log({"lr_step": scheduler.get_last_lr()[0]})

                running_loss_train += loss.item()  # save current loss to compute later a mean

                _, predicted = torch.max(outputs.data, 1)  # compute max logits, along dim 1, return (max, id_max==label)
                
                train_total += labels.size(0)  # how much samples seen so far
                train_correct += (predicted == labels).sum().item()  # how much corrected seen so far

                train_y_pred = torch.cat((train_y_pred, predicted.view(predicted.shape[0]).cpu()))
                train_y_true = torch.cat((train_y_true, labels.view(labels.shape[0]).cpu()))
                
            end = time.time()

            train_acc = 100*train_correct/train_total
            train_f1 = f1_score(train_y_true, train_y_pred, average='macro')
            
           
            self.net.eval()  # switch net to evaluate setting 

                
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                 for inputs, labels in tqdm(eval_loader, total=len(eval_loader), desc='Val round', unit='batch', leave=False):   # for each batch
                    eye, gsr, eeg, ecg = inputs  # Tensors

                    if self.use_cuda:
                        eye, gsr, eeg, ecg = eye.cuda('cuda:%i' %self.gpu_num), gsr.cuda('cuda:%i' %self.gpu_num), eeg.cuda('cuda:%i' %self.gpu_num), ecg.cuda('cuda:%i' %self.gpu_num), 
                        labels = labels.cuda('cuda:%i' %self.gpu_num)
                        
                    eval_outputs = self.net(eye, gsr, eeg, ecg)
                    eval_loss = self.val_criterion(eval_outputs, labels)
                    running_loss_eval += eval_loss.item()  # save current loss to compute later a mean

                    _, predicted = torch.max(eval_outputs.data, 1)  # compute max logits, along dim 1, return (max, id_max==label)
                    
                    total += labels.size(0)  # how much samples seen so far
                    correct += (predicted == labels).sum().item()  # how much corrected seen so far

                    y_pred = torch.cat((y_pred, predicted.view(predicted.shape[0]).cpu()))
                    y_true = torch.cat((y_true, labels.view(labels.shape[0]).cpu()))

            acc = 100*correct/total
            f1 = f1_score(y_true, y_pred, average='macro')

            # Log metrics
            wandb.log({"train loss": running_loss_train/len(train_loader), "train acc": train_acc, "train f1": train_f1,
                           "val loss": running_loss_eval/len(eval_loader), "val acc": acc, "val f1": f1, "lr": scheduler.get_last_lr()[0],
                           "epoch": epoch+1})

            print('Epoch {:03d}: Loss {:.4f}, Accuracy {:.4f}, F1 score {:.4f} || Val Loss {:.4f}, Val Accuracy {:.4f}, Val F1 score {:.4f}  [Time: {:.4f}]'
                  .format(epoch + 1, running_loss_train/len(train_loader), train_acc, train_f1, running_loss_eval/len(eval_loader), acc, f1, end-start))
            
            if f1 > best_f1:
                best_f1 = f1
                best_loss = running_loss_eval/len(eval_loader)
                best_acc = acc

            # Early stopping
            if self.es_mode == 'max':
                early_stopping(f1, self.net)
            else:
                early_stopping(running_loss_eval/len(eval_loader), self.net)
            if early_stopping.early_stop:
                print(f"Early stopping")
                break
            
        print(f'Finished Training')

        wandb.log({"Best val loss": best_loss})
        wandb.log({"Best val acc": best_acc})
        wandb.log({"Best val f1": best_f1})