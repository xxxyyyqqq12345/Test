function out=compute_out(input,W,skip,bias,ststr)
if length(skip)==1
  skip_1=skip_2=skip;
  
else
  skip_1=skip(1);
  skip_2=skip(2);
end
if length(bias)<size(W,4)
  len=size(W,4)-length(bias);
  Bias=bias;
  for i=1:len
    Bias(i+length(bias))=bias(1);
  end
  
else
  Bias=bias;

end

#out=0;
s=size(W);
if length(s)<4
  s(4)=1;
end
if s(3)==0
  s(3)=1;
end

I=0;
J=0;
for l=1:s(4)
  #I=0;
  #J=0;
  
  if isequal(ststr,"valid")
    S=convn(input,W(:,:,:,l),ststr);
  elseif isequal(ststr,"full")
    S=convn(input(:,:,1),W(:,:,1,l),ststr);
    for i=2:s(3)
      S=S+convn(input(:,:,i),W(:,:,i,l),ststr);
    end
  end
%  for i=1:skip_1:size(input)(1)-s(1)+1
%    I=I+1;
%    J=0;
%    for j=1:skip_2:size(input)(2)-s(2)+1
%      J=J+1;
%      M=input(i:i+s(1)-1,j:j+s(2)-1,:);
%      S=sum((M.*W(:,:,:,l))(:));
%      out(I,J,l)=S+bias(l);
%    end
%  end
  B(:,:,:,l)=S+Bias(l);
end
for i=1:skip_1:size(B(:,:,1),1)
  I=I+1;
  J=0;
  for j=1:skip_2:size(B(:,:,1),2)
    J=J+1;
    out(I,J,:)=B(i,j,:);
  end
end
end