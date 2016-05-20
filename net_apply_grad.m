function net=net_apply_grad(cnn,x,y,opts)
    m = size(x, 4);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        #for l = 1 : numbatches
        #batch_x = x(:, :, :,kk((i - 1) * opts.batchsize + 1 : l * opts.batchsize));
        #batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        net=cnn;
        Output=net_ff_prop(cnn,x);
        [delta,del,bias_deriv]=net_bb_alg(cnn,x,y,Output);
        net = net_grad(cnn, del,bias_deriv,opts);
        #if isempty(net.rL)
        #    net.rL(1) = net.L;
        #end
        #net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        #end
        toc;
    end
    

end