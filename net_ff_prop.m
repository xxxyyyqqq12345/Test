function Output=net_ff_prop(net,x)
Output{1}.Out=x;

for i=2:length(net.layers)
  if strcmp(net.layers{i}.type, 'c')
    size_W=size(net.layers{i}.W);
    #for l=1:size_W(4)
    Output{i}.Z=compute_out(Output{i-1}.Out,net.layers{i}.W,net.layers{i}.skip,net.layers{i}.bias,"valid");
    Output{i}.Out=sigm(Output{i}.Z);
    #end
  end
  if strcmp(net.layers{i}.type, 's')
    dim=net.layers{i}.dim; ##<----Here May change
    [Output{i}.Out,Output{i}.M]=reduce_dim(Output{i-1}.Out,dim);
  end
  input=Output{i};
end
n_layers=numel(Output);
Output{n_layers+1}.Z=net.ffW*Output{n_layers}.Out(:)+net.ffWB;

Output{n_layers+1}.Out=sigm(Output{n_layers+1}.Z);
end