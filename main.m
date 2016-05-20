#main.m
clear;clc;
addpath("util");
#batch_x = rand(28,28,5);
#batch_y = rand(10,5);
a=[256,256,3,1];
cnn.layers = {
    struct('type', 'i',"size",a) %input layer
    #struct('type', 's','dim', [1,1])
    struct('type', 'c', 'outputmaps', 2, 'size', [5,5,3,2],"skip",[1,1],"pad",0) %convolution layer
    struct('type', 's','dim', [2,2]) %sub sampling layer
    struct('type', 'c', 'outputmaps', 2, 'size', [2,2,2,3],"skip",[1,1],"pad",0) %convolution layer
    struct('type', 's', 'dim', [5,5]) %subsampling layer
    struct('type', 'c', 'outputmaps', 2, 'size', [3,3,3,1],"skip",[1,1],"pad",0) %convolution layer
    struct('type', 's', 'dim', [1,1]) %subsampling layer
    struct('type', 'c', 'outputmaps', 2, 'size', [5,5,1,1],"skip",[1,1],"pad",0) %convolution layer
    struct('type', 's', 'dim', [1,1]) %subsampling layer
    struct('type', 'c', 'outputmaps', 2, 'size', [8,8,1,1],"skip",[1,1],"pad",0)
    struct('type', 's', 'dim', [1,1]) %subsampling layer
    struct('type', 'c', 'outputmaps', 2, 'size', [8,8,1,1],"skip",[1,1],"pad",0)
    struct('type', 's', 'dim', [1,1]) %subsampling layer
    #struct('type', 'c', 'outputmaps', 2, 'size', [8,8,1,1],"skip",[1,1],"pad",0)
    #struct('type', 's', 'dim', [1,1]) %subsampling layer
    #struct('type', 'c', 'outputmaps', 2, 'size', [8,8,1,1],"skip",[1,1],"pad",0)
    #struct('type', 's', 'dim', [1,1]) %subsampling layer
    #struct('type', 'c', 'outputmaps', 2, 'size', [2,2,1,1],"skip",[1,1],"pad",0)
    #struct('type', 's', 'dim', [2,2]) %subsampling layer
    #struct('type', 's', 'dim', [2,2]) %subsampling layer
    #struct('type', 'c', 'outputmaps', 2, 'size', [2,2,1,1],"skip",[1,1],"pad",0) %convolution layer
    #struct('type', 's', 'dim', [1,1]) %subsampling layer
    #struct('type', 'c', 'outputmaps', 2, 'size', [2,2,1,1],"skip",[1,1],"pad",0)
};
#cnn.pad=0;
net=cnn;
x=rand(a);
y=rand(1,5);
#y=rand();
net = net_init(net,[2,2],y);
#cnn = cnnsetup(cnn, batch_x, batch_y);

opts.numepochs=1;
opts.batchsize=1;
opts.alpha=1;
Output=net_ff_prop(net,x);
[delta,del,bias_deriv]=net_bb_alg(net,x,y,Output);

for i=1:150
  net=net_apply_grad(net,x,y,opts);
  Output=net_ff_prop(net,x);
  n=numel(Output);
  X(:,:,i)=Output{n}.Out;
  X1(:,:,i)=Output{n}.Z;
  opts.alpha=opts.alpha./2;
  opts.alpha=max(opts.alpha,0.1);
end
y
[delta,del,bias_deriv]=net_bb_alg(net,x,y,Output);
#X(:,:,i)
#plot(max(X))
#plot(X1)

#cnn=cnntrain(cnn, batch_x, batch_y, opts);