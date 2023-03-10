import os,sys,time
import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../')
from networks.quant_layer import Linear_Q, Conv2d_Q


class Appr(object):

    def __init__(self,model,args,lr_patience=5,lr_factor=3,lr_min=1e-4):
        self.model=model
        self.device = args.device

        self.init_lr=args.lr
        self.momentum=args.momentum
        self.weight_decay=args.weight_decay

        # patience decay parameters
        self.lr_patience = lr_patience
        self.lr_factor=lr_factor
        self.lr_min=lr_min

        self.sbatch=args.sbatch
        self.nepochs=args.nepochs

        self.output=args.output
        self.checkpoint = args.checkpoint
        self.experiment=args.experiment
        self.num_tasks=args.num_tasks

        self.criterion = nn.CrossEntropyLoss()

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.init_lr
        return torch.optim.SGD(self.model.parameters(),lr=lr,momentum=self.momentum, weight_decay=self.weight_decay)

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        self.optimizer = self._get_optimizer()

        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf
        patience = self.lr_patience
        lr=self.init_lr

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(t,xtrain,ytrain)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain)
                clock2=time.time()

                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),
                    train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("saved best model and quit because loss became nan")
                    break

                if valid_loss<best_loss:
                    best_loss=valid_loss
                    patience=self.lr_patience
                    best_model=copy.deepcopy(self.model.state_dict())
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr)

                print()
        except KeyboardInterrupt:
            print()

        # Restore best
        self.model.load_state_dict(copy.deepcopy(best_model))

    def train_epoch(self,t,x,y):

        self.model.train()
         
        r=np.arange(x.size(0))
        print("-------------------")
        print(r.shape)
        print(x.shape)
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(self.device)

        # Loop batches
        for i in range(0,len(r),self.sbatch):

            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images, targets = x[b].to(self.device), y[b].to(self.device)

            # Forward
            outputs=self.model(images)
            outputs_t=outputs[t]
            loss = self.criterion(outputs_t, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Update parameters
            self.optimizer.step()

            # clipping parameters as mentioned in paper
            for m in self.model.modules():
                if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                    m.clipping()
        return

    def eval(self,t,x,y,debug=False):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.as_tensor(r, device=self.device, dtype=torch.int64)

        with torch.no_grad():
            # Loop batches
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images, targets = x[b].to(self.device), y[b].to(self.device)

                # Forward
                outputs=self.model(images)
                outputs_t=outputs[t]
                loss = self.criterion(outputs_t, targets)

                _,pred=outputs_t.max(1, keepdim=True)

                total_loss += loss.detach()*len(b)
                total_acc += pred.eq(targets.view_as(pred)).sum().item() 
                total_num += len(b)

        return total_loss/total_num, total_acc/total_num

    def save_model(self,t):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))
