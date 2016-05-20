function M=flip_matrix(W)
M=zeros(size(W,1),size(W,2),size(W,4),size(W,3));
for i=1:size(W,4)
  for j=1:size(W,3)
    M(:,:,i,j)=W(:,:,j,i)';
  end
  
end
end