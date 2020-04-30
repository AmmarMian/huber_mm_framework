function [Xnew, sig1, X1supp] = hubsniht(Y,A,k,X0supp,qn,printitn)
% 
% Multichannel sparse recovery of complex-valued signals using Huber's 
% criterion for joint estimateion of regression and scale. 
% The algorithm is based on greedy pursuit Simultaneous Normalized Iterative 
% Hard Thresholding (SNIHT) algorihtm 
% 
%  INPUT  
%        Y   :=  m x q matrix of measurements 
%        A   :=  m x n measurement matrix 
%        k   :=  the number of nonzero rows in signal matrix  
%        qn  :=  quantile value that determines the threshold  c of complex 
%                Huber's loss function  (Default qn = 0.8)                      
%        printitn := print iteration number    
% OUTPUT  
%        Xnew    := estimated n x q signal matrix with exactly k nonzero rows
%        X1supp  := estimated support (index) set of nonzeros rows
%
% Author: Esa Ollila, Jan 20th, 2015 
% sent comments to firstname.lastname@aalto.fi
%
% If you use this code, please cite the paper: 
%
% E. Ollila, "Multichannel sparse recovery of complex-valued signals using 
% Huber's criterion", in Proc. CoSeRa'15, Pisa, Italy, June 16-19, 2015. 
% [ArXiv identifier :  arXiv:1504.04184]
%--------------------------------------------------------------------------


[m, n] = size(A);
[~, q] = size(Y);

% if isreal(Y), 
%     error('hubsniht -> error: this function is for complex-valued measurements'); 
% end

debug = 0; % put 1 if you want the function to print out extra statistics 
% for debuggin purposes 

if nargin < 6, printitn=0; end
if nargin < 5, qn=0.8; end
if nargin < 4, X0supp=[]; end

if k > n, 
    error('hubsniht -> error: number of nonzeros k=%d larger than n = %d',k,n); 
end

%%- Function declarations for Huber
rhofun = @(x,c) (abs(x).^2).*(abs(x)<=c) + (2*c*abs(x)-c^2).*(abs(x)>c);
psifun = @(x,c) ( x.*(abs(x)<=c) + c*sign(x).*(abs(x)>c));
wfun =  @(x,c) ( 1.*(x<=c) + (c*(1./x)).*(x>c) );
vec = @(x) (x(:));

% For a given qn, compute beta (concistency factor)
csq = chi2inv(qn,2)/2;  % threshold c^2
c = sqrt(csq);
alpha = chi2cdf(2*csq,4)+csq*(1-qn); % consistency factor for scale 

%%-- initial approximation (is the zero matrix) 
X0  = zeros(n,q); 
sig0 = 1.20112*median(abs(Y(:))); % initial scale statistic 

%% TO DO: 
% DOA =  true; %false; % true if the application is DOA finding, then we  
% locate the peaks instead of finding the k largest 'correlations'

%%- DetectSupport
if isempty(X0supp),
    
    Ypsi = psifun(Y/sig0,c)*sig0; % pseudo 'residuals' 
    R = A'*Ypsi;  
    [~, indx] = sort(sum(R.*conj(R),2),'descend');
    X0supp = indx(1:k);
    
    %% TO DO (I will add this functionality later)  
    %if DOA, % We do not want adjacent DOA's but to locate the peaks  
    %   X0supp = locatepeaks(R,indx) 
    %end
end
    
ITERMAX = 500;
muold=0;
ERRORTOL = 1e-8; % ERROR TOLERANCE FOR HALTING CRITERION

%% uncomment if you wish to do extra computations for debugging
% if debug, 
%    objold  = 1e12; % a big number
% end


for iter = 1:ITERMAX, 
    
    %%-- Compute the negative gradient 
    R = Y - A*X0;
    
    %%-- Scale step
    sig1sq = (sig0^2/(alpha*(m*q)))*sum(sum(abs(psifun(R/sig0,c)).^2));
    sig1 = sqrt(sig1sq);
     
    %%- Negative of the gradient 
    psires = psifun(R/sig1,c)*sig1;  
    G = A'*psires;

    %%-- stepsize computation 
    tildeR = R - muold*A(:,X0supp)*G(X0supp,:);
    W  = wfun(abs(tildeR)/sig1,c);        % weights
    W(tildeR==0) = 0;

    tmp = vec(A(:,X0supp)*G(X0supp,:));
    mu2 = sum((tmp.*conj(tmp)).*vec(W));
    mu1 = (tmp.*vec(W))'*vec(R);
    mu =  real(mu1)/mu2;
   
    %%-- Next proposal for signal matrix X 
    X1 = X0 + mu*G;
    
    %%-- Detect Support
    [~, indx] = sort(sum(X1.*conj(X1),2),'descend');
    X1supp = indx(1:k);
    
    %%-- Hard-thresholding 
    Xnew   = zeros(n,q);
    Xnew(X1supp,:)= X1(X1supp,:);
    
    %%-- Stopping criteria          
    crit = norm(Xnew-X0,'fro')^2/norm(Xnew,'fro')^2;
    
    if mod(iter,printitn) == 0,
       fprintf('%3d: mu = %.4f crit = %.9f \n',iter,mu,crit);
    end
      
    %% Uncomment these for ONLY debugging the code  
    %% these stuff compute the value of objective functions and estimating eq's 
    %if debug, 
    %   R = (Y - A*Xnew)/sig1;
    %   eq1 = A(:,X1supp)'*psifun(R,c);
    %   eq2 = sum(sum(abs(psifun(R,c)).^2));
    %   objnew = sig1*sum(sum(rhofun(R,c))) + alpha*m*q*sig1;
    %   if mod(iter,printitn) == 0,
    %       fprintf('  obj = %.3f eq1 = %.4f\n',objnew,norm(eq1,'fro'));
    %       fprintf('  eq2 = %.5f\n',abs(eq2-alpha*m*q));
    %   end
    %   if objnew > objold,  
    %        fprintf('increasing objective fnc'); 
    %   end
    %end
    
    if crit < ERRORTOL, 
      break;  
    end 
        
    muold = mu;
    X0     = Xnew;
    X0supp = X1supp;
    sig0   = sig1; 
    %objold = objnew; % uncomment when debugging 
        
end

if printitn, 
    fprintf('hubsiniht --> terminating at iter = %d crit = %f\n',iter,crit); 
end

X1supp = sort(X1supp);