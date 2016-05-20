function [delta,del,bias_deriv]=net_bb_alg(net,x,y,Output)

n=numel(Output);
delta{n+1}=Output{n}.Out'-y;
#C = 1/2* sum(delta{1} .^ 2) / size(delta{n+1}, 2);

%%  backprop deltas
#Out_d = diff .* (Out{n} .* (1 - Out{n}));   #real diff
#fvd = (net.ffW * Out_d); #deltalast
#temp_s=size(Out{n-1});
#delta{n+1} = delta{n+1}';
delta{n}=delta{n+1}*net.ffW.*(Output{n-1}.Out(:) .* (1 - Output{n-1}.Out(:)))';
#delta(n-2)=compute_out(delta{n-1},net.layers{i+1}.W,[1,1],0);

bias_deriv{n}=0;

#del{n}=0;
del{n}=(Output{n-1}.Out(:)*delta{n+1})';


delta{n-1} = reshape(delta{n}, size(Output{n-1}.Out,1), size(Output{n-1}.Out,2), size(Output{n-1}.Out,3));
#bias_deriv{n-1}=;
for i=(n - 2) : -1 : 1
   if strcmp(net.layers{i}.type, 'c')
    clear temp;
    for ii=1:size(delta{i+1},3)
      temp(:,:,ii)=expand(delta{i+1}(:,:,ii),net.layers{i+1}.dim);#Output{i}.Out.*(1-Output{i}.Out).*Output{i+1}.M;
    end
    delta{i}=temp.*Output{i+1}.M;
   elseif strcmp(net.layers{i}.type, 's')
     delta{i}=compute_out(delta{i+1},flip_matrix(flipdim(flipdim(net.layers{i+1}.W, 1), 2)),net.layers{i+1}.skip,0,"full");
     #delta{i}=compute_out(delta{i+1},flip_matrix(net.layers{i+1}.W),net.layers{i+1}.skip,0,"full");

   end
  
  
end

for l = 2 : n-1
    if strcmp(net.layers{l}.type, 'c')
      for i=1:size(Output{l-1}.Out,3)
            del{l}(:,:,:,i) =compute_out(flipall(Output{l-1}.Out(:,:,i)), flip_matrix(delta{l}),net.layers{l}.skip,0, 'valid');# / size(net.layers{l}.d{j}, 3);
            #bias_deriv{l}(i) = sum(delta{l}(:,:,i)(:));#size(net.layers{l}.d{j}, 3);
      end
      del{l}=flip_matrix(del{l});
      for j=1:size(Output{l}.Out,3)
        bias_deriv{l}(j) = sum(delta{l}(:,:,j)(:));#size(net.layers{l}.d{j}, 3);
      end
    end
end


bias_deriv{n+1}= sum(delta{n+1});

%fv=[];
%for l=1:temp_s(3)
%  fv=[fv;Out{n-1}(:,:,l)];
%end
%
%OUT{1}=fv;
%if strcmp(net.layers{n}.type, 'c') 
%   fvd = fvd .* (OUT{1} .* (1 - OUT{1}));
%end
%ind=1;
%for i=(n - 1) : -1 : 1
%  if strcmp(net.layers{n}.type, 'c') 
%    fvd = fvd .* (OUT{ind} .* (1 - OUT{ind}));
%  end
%  
%end
%if strcmp(net.layers{n}.type, 'c') 
%    fvd = fvd .* (fv .* (1 - fv));
%end
%
%end