# coding: utf-8

#单张图片测试请直接更改img_path，整个测试集图片测试请使用整体测试部分代码

import mxnet as mx
import numpy as np
import cv2

#initial
# 测试图片路径,请在此处添加测试图片路径
img_path = 'C:/Users/Administrator/Desktop/aa.jpg'
prefix = 'face'
batch_size = 10
device = mx.cpu()
input_shape = (112,92)

#整体测试
#封装成rec文件
#test_dataiter = mx.io.ImageRecordIter(
#        path_imgrec="./testdata/testdata_test.rec",
#        path_imglist="./testdata/testdata_test.lst",
#        data_shape=(3,112,92),
#        batch_size=batch_size,
#        rand_crop=True,
#        label_width=4,
#        )
#sym,arg_params,aux_params = mx.model.load_checkpoint(prefix=prefix,epoch=100)
#mod = mx.mod.Module(symbol=sym,data_names=('data',),context=device)
#mod.bind(for_training=False,data_shapes=[('data',(batch_size,3,112,92))])
#mod.set_params(arg_params,aux_params,allow_missing=True)
#batchX = test_dataiter
#y = mod.predict(batchX)
#mod.forward(batchX.getdata().asnumpy(), is_train=False)
#gender = y[0].asnumpy()
#glasses = y[1].asnumpy()
#masks = y[2].asnumpy()
#hat = y[3].asnumpy()
#list1 = ['男性','女性']
#list2 = ['没戴眼镜','透明眼镜','墨镜']
#list3 = ['没戴口罩','黑色口罩','白色口罩','其它颜色口罩']
#list4 = ['没带帽子','带了帽子']
#for i in range(y[0].shape[0]):
#    print '第%d个人为%s,%s,%s,%s'%(i,list1[gender[i].argmax()],list2[glasses[i].argmax()],list3[masks[i].argmax()],list4[hat[i].argmax()])

#单张图片测试
def adjust_input(in_data):
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data
    out_data = out_data.transpose((2,0,1))
    out_data = np.expand_dims(out_data,0)
    return out_data
img = cv2.imread(img_path)
cv2.imshow('原始图片',img)
[r,g,b] = cv2.split(img)
img = cv2.merge((b,g,r))
im_data = cv2.resize(img,input_shape)
input_buf = adjust_input(im_data)
mod = mx.model.FeedForward.load(prefix=prefix, epoch=100, ctx=device)
y = mod.predict(input_buf)
gender = y[0]
glasses = y[1]
masks = y[2]
hat = y[3]

# print result
list1 = ['男性','女性']
list2 = ['没戴眼镜','透明眼镜','墨镜']
list3 = ['没戴口罩','黑色口罩','白色口罩','其它颜色口罩']
list4 = ['没带帽子','带了帽子']
print '编号为%s的人为%s,%s,%s,%s'%(img_path[-9:-4],list1[gender.argmax()],list2[glasses.argmax()],list3[masks.argmax()],list4[hat.argmax()])
cv2.waitKey(10000)
