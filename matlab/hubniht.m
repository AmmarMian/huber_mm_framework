function [b1, sig1, supp1,iter,failure] = hubniht(y,X,K,supp0,sig0,b0,c,printitn)
% 
% Normalized Iterative Hard Thresholding (NIHT) algorihtm of real-valued 
% signals using Huber's criterion for joint estimation of regression and
% scale. 
% 
%  INPUT  
%         y: N x 1 matrix of measurements 
%         X: N x p measurement matrix 
%         K: natural number, stating the number of nonzeros   
%     supp0: initial estimate of support 
%         c: threshold for Huber's criterion               
%  printitn: print iteration number    
%
% OUTPUT  
%        b1: estimated px1 signal vector with exactly k nonzero elements
%     supp1: estimated support, i.e., the index set of nonzeros elements
%      sig1: the estimate of scale 
%
% Author: Esa Ollila, Aalto University, 2020
%--------------------------------------------------------------------------

[N, p] = size(X);

if nargin < 8
    printitn = 1; 
end

if nargin < 7 || isempty(c)
   c = 1.3415;  % Default: approx 95 efficiency for Gaussian errors
end

if nargin < 6 || isempty(b0)
    b0 = zeros(p,1); 
end
r = y - X*b0;

if nargin < 5|| isempty(sig0)
    sig0 = 1.4826*median(abs(r)); % initial scale statistic
end

if nargin < 4 || isempty(supp0)
    supp0 = []; 
end

if K > N 
    error('hubniht: nr of nonzeros K=%d larger than N = %d',K,N); 
end

csq = c^2; 
% compute $\alpha$: consistency factor for scale 
al = (1/2)*chi2cdf(csq,3)+ (csq/2)*(1-chi2cdf(csq,1));
  
%%  Detect the support
if isempty(supp0)
    
    ypsi = psihub(r/sig0,c)*sig0; % winsorized observations 
    delta = X'*ypsi;  % correlations 
    [~, indx] = sort(abs(delta),'descend');
    supp0 = indx(1:K);
   
end
  
%% initialize 
ITERMAX = 200;
ERRORTOL = 1.0e-4; % ERROR TOLERANCE FOR HALTING CRITERION

%% uncomment if you wish to do extra computations for debugging
con =  sqrt((N-K)*2*al);
objold  = 1e12; % a big number
crit1 = 1e12;
crit2 = 1e12;
mu0  = 0;
lam0 = 0;
failure = false;

for iter = 1:ITERMAX 
        
    % STEP 2: update $\tau$
    tau = (norm(psihub(r/sig0,c))/con);
    
    % STEP 3: update step size for scale 
    % update only when crit 1 > 0.001
    if crit1 > 0.001 && iter > 4
        lam_num = norm(psihub(r/(sig0*(tau^lam0)),c))/con;
        lam = lam0 + log(lam_num)/log(tau);  
        lam = max(0.01,min(lam,1.99));
    else
        lam = lam0;
    end
    %lam = 1;
    if iter < 4,  lam = 1; end

    % STEP 4: update the scale 
    update1 = tau^lam;
    sig1 = sig0*update1;

    % STEP 5: Update $\delta$  
    psires = psihub(r/sig1,c)*sig1;  
    delta = X'*psires;
    
    % STEP 6: update the step size of regression
    % update only when crit 1 > 0.001
    if crit2 > 0.001 && iter > 4
        z = X(:,supp0)*delta(supp0);
        tilde_r = r - mu0*z;
        w  = whub(abs(tilde_r)/sig1,c);        % weights
        w(tilde_r==0) = 0.000001;
        mu2 = sum((z.^2).*w);
        mu1 = sum((z.*w).*r);
        mu =  max(0.01,min(mu1/mu2,1.99));
    else
        mu = mu0;
    end
    
    if iter < 4 ,  mu = 1; end
    %mu = 1;
    update2  = mu*delta;

    % STEP 7: update the regression vector:
    b1tmp = b0 + update2; 
    [~, indx] = sort(abs(b1tmp),'descend');
    supp1 = indx(1:K);
    b1    = zeros(p,1);
    b1(supp1) = b1tmp(supp1,:);
      
    r = y - X*b1;
    objnew = sig1*sum(rhohub(r/sig1,c))/(N-K) + al*sig1;

    while objnew > objold && iter > 4
         
        mu = mu/2; 
 
        % Update the proposal for b 
        b1tmp = b0 + mu*delta;
        [~, indx] = sort(abs(b1tmp),'descend');
        supp1 = indx(1:K);
        b1    = zeros(p,1);
        b1(supp1) = b1tmp(supp1,:);
         
        % Update objnew
        r = y - X(:,supp1)*b1(supp1);

        objnew = sig1*sum(rhohub(r/sig1,c))/(N-K) + al*sig1;
         
        if mu < 0.001
            break;
        end
    end
    
    % STEP 8: stopping criteria
    crit2 = norm(b1-b0)/norm(b1);
    crit1 = abs(update1 - 1);     
        
    if mod(iter,printitn) == 0
        
        tmp2 = psihub(r/sig1,c)*sig1;     
        esteq_beta = X(:,supp1)'*tmp2/N;
        esteq_sigma =  sum(psihub(r/sig1,c).^2)/(N-K) - 2*al;
        
        fprintf('%2d %.3f|%.3f\t %.9f %.9f %11.7f  %10.5f  %10.5f\n', ... 
            iter,mu,lam,crit2,crit1,objnew,norm(esteq_beta),abs(esteq_sigma));
        if objnew > objold 
            fprintf('increasing objective fnc\n'); 
        end
    end
    
    if (crit2 < ERRORTOL  && crit1 < ERRORTOL)
        break;
    end 
        
    b0 = b1; 
    sig0 = sig1; 
    mu0 = mu;
    lam0 = lam;
    objold = objnew; 
    supp0 = supp1;
    
end

if mod(iter,printitn)==0
    fprintf('\n Done! \n');
    fprintf('hubniht : terminating at iter = %d crit2 = %f\n',iter,crit2); 
end

if iter == ITERMAX
    failure = true;
 %   fprintf('error!!! MAXiter = %d crit2 = %.7f\n',iter,crit2)
end

supp1 = sort(supp1);