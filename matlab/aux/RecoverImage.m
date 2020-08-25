function yout = RecoverImage(N,D,CoefMatrix,direction)
% INPUT
%   N           the size of the image. Image is assumed to be N x N image
%   D           n^2 x m dictionary, where n is the size of the image blocks 
%   CoefMatrix  sparsematrix of size n^2 x (# patches)
%   direction   (optional) sring either equal to 'horizontal' or 'vertical'
%               Default is horizontal
%
% OUTPUT
%   yout         N X N image
%
% Modified from the codes of Michael Elad accompanying his book, "Sparse
% and redundant representations". 
%--------------------------------------------------------------------------

n = sqrt(size(D,1)); 
yout = zeros(N,N); 

i=1; 
j=1;
patches_dn = D*CoefMatrix;

if nargin < 4
    direction = 'horizontal';
end

assert(any(strcmpi(direction,{'horizontal','vertical'})),['''direction'' must' ...
    ' be a string equal to ''horizontal'' or ''vertical''']);

if strcmpi(direction,'horizontal')
    
    for k=1:1:(N-n+1)^2
        
        patch = reshape(patches_dn(:,k),[n,n]);
        yout(i:i+n-1,j:j+n-1) = yout(i:i+n-1,j:j+n-1) + patch; 
        if j < N-n+1 
            j = j+1; 
        else
            j=1; 
            i = i + 1; 
        end
    end
    
else
    
    for k=1:1:(N-n+1)^2
        patch = reshape(patches_dn(:,k),[n,n]);
        yout(i:i+n-1,j:j+n-1) = yout(i:i+n-1,j:j+n-1) + patch; 
        if i < N-n+1 
            i = i+1; 
        else
            i=1; 
            j = j + 1; 
        end
    end 

end


