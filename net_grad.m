function net=net_grad(cnn, del,bias_deriv,opts)
    for l = 2 : numel(cnn.layers)
        if strcmp(cnn.layers{l}.type, 'c')
            cnn.layers{l}.W=cnn.layers{l}.W-opts.alpha*del{l};
            cnn.layers{l}.bias=cnn.layers{l}.bias-opts.alpha*bias_deriv{l};
        end
    end
    n=numel(cnn.layers);
    cnn.ffW = cnn.ffW - opts.alpha * del{n+1};
    cnn.ffWB = cnn.ffWB - opts.alpha * bias_deriv{n+2};
    net=cnn;
end