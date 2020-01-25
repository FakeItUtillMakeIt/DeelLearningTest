#VGG

import d2lzh as d2l
from mxnet import gluon,init,nd
from mxnet.gluon import nn

#vgg块使用多个相同的卷积层，最后通过一个池化大小为2，步幅为2的池化层输出
def vgg_block(num_conv,num_channels):
    blk=nn.Sequential()
    for i in range(num_conv):
        blk.add(nn.Conv2D(num_channels,kernel_size=3,padding=1,activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2,strides=2))
    return blk
#conv_arch存储vgg网络结构
conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512))

def vgg_net(conv_arch):
    net=nn.Sequential()
    for (conv,channels) in conv_arch:
        net.add(vgg_block(conv,channels))
        #全连接层
    net.add(nn.Dense(4096,activation='relu')),nn.Dropout(0.5)
    net.add(nn.Dense(4096,activation='relu')),nn.Dropout(0.5)
    net.add(nn.Dense(10))
    return  net

# vgg_net=vgg_net(conv_arch)

# vgg_net.initialize()
# X=nd.random.uniform(shape=(1,1,224,224))
#
# for net in vgg_net:
#     X=net(X)
#     print(net.name,X.shape)

ratio=4
small_conv_arch=[(pair[0],pair[1]//ratio) for pair in conv_arch]
vgg_net=vgg_net(small_conv_arch)

lr,num_epochs,batch_size,ctx=0.05,5,128,d2l.try_gpu()
vgg_net.initialize(ctx=ctx,init=init.Xavier())#初始化权值参数
trainer=gluon.Trainer(vgg_net.collect_params(),'sgd',{'learning_rate':lr})
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=224)
d2l.train_ch5(vgg_net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)

# def vgg_block(conv_param):
#     blk=nn.Sequential()
#     for i in range(conv_param[0]):
#         blk.add(nn.Conv2D(conv_param[1],kernel_size=3,padding=1,activation='relu'))
#     blk.add(nn.MaxPool2D(pool_size=2,strides=2))
#     return blk
#
#
# def vgg19_net():
#     net=nn.Sequential()
#     conv_arch19_net1 = (2,64)
#     conv_arch19_net2=(2,128)
#     conv_arch19_net3=(4,256)
#     conv_arch19_net4=(4,512)
#     conv_arch19_net5=(4, 512)
#     net.add(
#                vgg_block(conv_arch19_net1),
#                vgg_block(conv_arch19_net2),
#                vgg_block(conv_arch19_net3),
#                vgg_block(conv_arch19_net4),
#                vgg_block(conv_arch19_net5),)
#     net.add(
#             nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
#             nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
#             nn.Dense(10,activation='softrelu')
#             )
#     return net
#
# #out of memory
# vgg_net19=vgg19_net()
# lr,num_epochs,batch_size,ctx=0.05,5,128,d2l.try_gpu()
# vgg_net19.initialize(init=init.Xavier(),ctx=ctx)
# trainer=gluon.Trainer(vgg_net19.collect_params(),'sgd',optimizer_params={'learning_rate':lr})
# train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=(224,224))
# d2l.train_ch5(vgg_net19,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)