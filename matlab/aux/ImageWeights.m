function Weight = ImageWeights(n,N)
% returns the N x N weight matrix for an N x N image that is denoised in 
% n by n patches using HUBNIHT_denoise function. 
%
% INPUT:
%   n = patch size (n by n)
%   N = image size (image is square N by N) 
% OUTPUT
%  W  = weight matrix of size N by N 
% 
% Author: Esa Ollila, 2020
%--------------------------------------------------------------------------
trim1 = (n+1):N-n;
trim2 = (n+1):N-n;
Weight = zeros(N,N);
Weight(trim1,trim2) = n^2;
for ii=1:n
    Weight(ii,:)=[ii:ii:ii*n n*repmat(ii,1,N-2*n) ii*n:-ii:ii];
    Weight(N-ii+1,:)=[ii:ii:ii*n n*repmat(ii,1,N-2*n) ii*n:-ii:ii];
    Weight(:,ii) = Weight(ii,:)'; 
    Weight(:,N-ii+1) = Weight(ii,:)'; 
end