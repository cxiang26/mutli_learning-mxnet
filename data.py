import sys
import os
import mxnet as mx

def data_iterator(batch_size):
    train_dataiter = mx.io.ImageRecordIter(
        path_imgrec="./recdata/mydata_train.rec",
        #path_imglist="./recdata/mydata_train.lst",
        data_shape=(3,112,92),
        rand_crop=True,
        batch_size=batch_size,
        rand_mirror=True,
        shuffle=True,
        label_width = 4,
        round_batch=True
        )
    test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="./recdata/mydata_test.rec",
        path_imglist="./recdata/mydata_test.lst",
        data_shape=(3,112,92),
        batch_size=batch_size,
        rand_crop=True,
        label_width=4,
        )
    return (train_dataiter,test_dataiter)