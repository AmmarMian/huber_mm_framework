function X = SOMP2(A,Y,flavor,k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Aqib Ejaz, Aalto University
% 15 December 2014
%
% Function:
% Simultaneous Orthogonal Matching Pursuit (SOMP) algorithm for sparse
% signal recovery in multiple measurement vectors (MMV) model
%
% Inputs:
% A - measurement matrix
% Y - observed data matrix 
% flavor - a string which specifies the stopping criterion for SOMP
% algorithm. flavor='sparsity' means SOMP should stop when the size of the
% support of the reconstructed matrix becomes equal to the input k. All
% other values of flavor mean the stopping criterion is based on the norm
% of the residual error
% k - either an integer that specifies the size of the support of the
% reconstructed matrix or a positive real number that specifies the norm 
% of the residual error
%
% Outputs:
% X - row sparse matrix which is recovered by SOMP 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a=diag(A'*A);
[~,n]=size(A);
[~,T]=size(Y);
X=zeros(n,T);
R=Y-A*X;

S=0;
i=1;
while(1)
    %correlate the current residue vectors in R with the columns of A
    z=sum((A'*R).^2,2)./a;
    
    %select the column with the maximum correlation and add its index to a
    %list of selected columns
    j=find(z==max(z),1,'first');
    S(i)=j;
    i=i+1;
    
    %project the observed vectors in Y to the subspace spanned by the
    %columns in the list to get current estimate of the sparse matrix
    V=A(:,S);
    try
        B=V\Y;
    catch
        B=pinv(V)*Y;
    end
    X=zeros(n,T);
    X(S,:)=B;
    
    %compute residue for the next iteration and see if the stopping
    %criterion is met
    R=Y-A*X;
    if (strcmp(flavor,'sparsity'))
        if (i>k)
            break;
        end
    else
        if (norm(R,'fro')<k || i>n)
            break;
        end
    end
end

