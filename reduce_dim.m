function [output,Ind]=reduce_dim(input,dim)
  assert(floor(size(input)([1,2])./dim)==size(input)([1,2])./dim, "dimension mismatch");
  Ind=zeros(size(input));
  I=J=L=0;
  for l=1:size(input,3)
    ++L;
    for i=1:dim(1):size(input)(1)
      I=I+1;
      for j=1:dim(2):size(input)(2)
        J=J+1;
        [output(I,J,L),In]=max(input(i:i+dim(1)-1,j:j+dim(2)-1,l)(:));
        Temp=Ind(i:i+dim(1)-1,j:j+dim(2)-1,l);
        Temp(In)=1;
        Ind(i:i+dim(1)-1,j:j+dim(2)-1,l)=Temp;
      end
      J=0;
    end
    I=0;
end