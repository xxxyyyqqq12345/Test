function net = net_init(net,dim,y_temp)
  #size=net.layers{1}.size;
  next_input_size=net.layers{1}.size;
  next_input_size
  for i=2:length(net.layers)
    if strcmp(net.layers{i}.type, 'c') || strcmp(net.layers{i}.type, 'rc')
      W_size=net.layers{i}.size;
      net.layers{i}.W=rand(W_size);
      if ~isfield(net.layers{i},"skip")
        net.layers{i}.skip=[1;1];
      end
      net.layers{i}.bias=rand(W_size(4),1);
      next_input_size([1,2])=(next_input_size([1,2])-size(net.layers{i}.W,[1,2])+2*net.layers{i}.pad)/net.layers{i}.skip+1;
      next_input_size(3)=size(net.layers{i}.W,4);
    elseif strcmp(net.layers{i}.type, 's')
      if !isfield(net.layers{i},"dim")
        net.layers{i}.dim=dim;
        next_input_size([1,2])=next_input_size([1,2])./dim;
      else
        next_input_size([1,2])=next_input_size([1,2])./net.layers{i}.dim;
      end
      
      
    end
    
    #assert(round(next_input_size)!=next_input_size);
    next_input_size

  end
  ffw_temp=rand(round(next_input_size))(:);
  net.ffW=rand(size(ffw_temp,1),size(y_temp,2));
  #net.ffW=rand(round(next_input_size))(:);
  net.ffW=net.ffW';
  net.ffWB=rand();
end
