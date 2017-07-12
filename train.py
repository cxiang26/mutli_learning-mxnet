from data import data_iterator
import mxnet as mx
import numpy as np
import logging
import time
logging.basicConfig(level=logging.DEBUG)

def build_network():
    #layer1
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, num_filter=32, kernel=(3,3), stride=(1,1), pad=(0,0), name='conv1')
    bn1 = mx.symbol.BatchNorm(data=conv1,name='bn1')
    act1 = mx.symbol.Activation(data=bn1,act_type='relu',name='act1')
    pool1 = mx.symbol.Pooling(data=act1,kernel=(2,2),stride=(2,2),pad=(0,0),pool_type='max',name='pool1')

    #layer2
    conv2 = mx.symbol.Convolution(data=pool1, num_filter=64, kernel=(3,3), stride=(1,1), pad=(0,0), name='conv2')
    bn2 = mx.symbol.BatchNorm(data=conv2,name='bn2')
    act2 = mx.symbol.Activation(data=bn2,act_type='relu',name='act2')
    pool2 = mx.symbol.Pooling(data=act2,kernel=(2,2),  stride=(2, 2), pad=(0, 0), pool_type='max', name='pool2')

    #layer3
    conv3 = mx.symbol.Convolution(data=pool2, num_filter=128, kernel=(3,3), stride=(1,1), pad=(0,0), name='conv3')
    bn3 = mx.symbol.BatchNorm(data=conv3,name='bn3')
    act3 = mx.symbol.Activation(data=bn3,act_type='relu',name='act3')
    pool3 = mx.symbol.Pooling( data=act3,kernel=(2,2), stride=(2,2),pad=(0,0),pool_type='max',name='pool3')

    #layer4
    conv4 = mx.symbol.Convolution(data=pool3, num_filter=128, kernel=(3,3), stride=(1,1), pad=(0,0), name='conv4')
    bn4 = mx.symbol.BatchNorm(data=conv4,name='bn4')
    act4 = mx.symbol.Activation(data=bn4,act_type='relu',name='act4')

    #layer5
    conv5 = mx.symbol.Convolution(data=act4, num_filter=256, kernel=(3,3), stride=(1,1), pad=(0,0), name='conv5')
    bn5 = mx.symbol.BatchNorm(data=conv5,name='bn5')
    act5 = mx.symbol.Activation(data=bn5,act_type='relu',name='act5')

    #layer6
    conv6 = mx.symbol.Convolution(data=act5, num_filter=256, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv6')
    bn6 = mx.symbol.BatchNorm(data=conv6,name='bn6')
    act6 = mx.symbol.Activation(data=bn6,act_type='relu',name='act6')

    flatten = mx.symbol.Flatten(data=act6)
    fc1 = mx.symbol.FullyConnected(data=flatten,num_hidden=500)
    act7 = mx.symbol.Activation(data=fc1,act_type='relu',name='act7')

    fc2_1 = mx.symbol.FullyConnected(data=act7,num_hidden=2)
    fc2_2 = mx.symbol.FullyConnected(data=act7,num_hidden=3)
    fc2_3 = mx.symbol.FullyConnected(data=act7,num_hidden=4)
    fc2_4 = mx.symbol.FullyConnected(data=act7,num_hidden=2)

    softmax2_1 = mx.symbol.SoftmaxOutput(data=fc2_1,name='softmax2_1')
    softmax2_2 = mx.symbol.SoftmaxOutput(data=fc2_2,name='softmax2_2')
    softmax2_3 = mx.symbol.SoftmaxOutput(data=fc2_3,name='softmax2_3')
    softmax2_4 = mx.symbol.SoftmaxOutput(data=fc2_4,name='softmax2_4')
    softmax = mx.sym.Group([softmax2_1,softmax2_2,softmax2_3,softmax2_4])
    return softmax

class Multi_iterator(mx.io.DataIter):
    def __init__(self,data_iter):
        super(Multi_iterator,self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data
    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0][1]
        return [('softmax2_1_label',(provide_label[0],)),\
                ('softmax2_2_label',(provide_label[0],)),\
                ('softmax2_3_label',(provide_label[0],)),\
                ('softmax2_4_label',(provide_label[0],))]
    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]
        return mx.io.DataBatch(data=batch.data,label=[label.T[0],label.T[1],label.T[2],label.T[3]],pad=batch.pad,index=batch.index)

class Multi_Accuracy(mx.metric.EvalMetric):
    def __init__(self,num=None):
        super(Multi_Accuracy,self).__init__('multi-accuracy',num)
    def update(self,labels,preds):
        mx.metric.check_label_shapes(labels,preds)

        if self.num != None:
            assert len(labels) == self.num
        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')
            mx.metric.check_label_shapes(label,pred_label)
            if i==None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

batch_size = 16
num_epochs = 100
device = mx.gpu(0)
lr = 0.001
profix='face'
checkpoint = mx.callback.do_checkpoint(profix,period=10)

network = build_network()
train,val = data_iterator(batch_size=batch_size)
train = Multi_iterator(train)
val = Multi_iterator(val)

model = mx.model.FeedForward(
    ctx=device,
    symbol=network,
    num_epoch=num_epochs,
    learning_rate=lr,
    momentum=0.9,
    wd=0.00001,
    initializer=mx.init.Xavier(factor_type='in',magnitude=2.34))

model.fit(
    X=train,
    eval_data = val,
    eval_metric=Multi_Accuracy(num=4),
    batch_end_callback=mx.callback.Speedometer(batch_size,32),
    epoch_end_callback=checkpoint)


