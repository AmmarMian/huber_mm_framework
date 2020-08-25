function A = redDCT(n)
% makes redundant 2D-DCT dictionary of size n x 256 atoms
% INPUT 
%    n = size of the image patch or blocks. For example, if you plan to
%           use 8 by 8 windows, then give n = 8. 
%    A = dictionary of size n^2 x 256
%
% Author: Esa Ollila, 2020. 
%--------------------------------------------------------------------------
K = 256; % number of atoms 
A = cos((0:1:n-1)'*(0:sqrt(K)-1)*pi/sqrt(K));
A(:,2:end) = A(:,2:end)-mean(A(:,2:end));
A = bsxfun(@rdivide, A, sqrt(sum(A.*A)));
A = kron(A,A); % 2D-DCT
A = bsxfun(@rdivide, A, sqrt(sum(A.*A)));