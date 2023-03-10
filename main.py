from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
import torch
import math
from networks.quant_layer import Linear_Q, Conv2d_Q
from approaches.blip_utils import estimate_fisher
import utils
numclass=10
feature_extractor=resnet18_cbam()
img_size=32
batch_size=128
task_size=10
memory_size=2000
epochs=1
learning_rate=2.0
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))
for i in range(10):
    task=i
    model.beforeTrain() 
    xtrain, ytrain, accuracy=model.train()

    xtrain=torch.from_numpy(xtrain).to(device)
    ytrain=torch.from_numpy(ytrain).to(device)
    estimate_fisher(task, model.model, xtrain, ytrain)

    for m in model.model.modules():
            if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                # update bits according to information gain
                m.update_bits(task=task, C=0.5/math.log(2))
                # do quantization
                m.sync_weight()
                # update Fisher in the buffer
                m.update_fisher(task=task)
    model.afterTrain(accuracy)

    utils.used_capacity(model, 20)

