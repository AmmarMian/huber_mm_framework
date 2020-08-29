function [b1, sig1, iter,mu] = hubreg(y,X,c,sig0,b0,printitn)
% [b1, sig1, iter] = hubreg(y,X,...)
% hubreg computes the joint M-estimates of regression and scale using 
% Huber's criterion. 
%
% INPUT: 
%
%        y: Numeric data vector of size N x 1 (output, respones)
%        X: Numeric data matrix of size N x p. Each row represents one 
%           observation, and each column represents one predictor (feature). 
%           If the model has an intercept, then first column needs to be a  
%           vector of ones. 
%         c: numeric threshold constant of Huber's function
%      sig0: (numeric) initial estimator of scale [default: SQRT(1/(n-p)*RSS)]
%        b0: initial estimator of regression (default: LSE)  
%  printitn: print iteration number (default = 0, no printing)
%
% OUTPUT:
%
%       b1: the regression coefficient vector estimate 
%     sig1: the estimate of scale 
%     iter: the # of iterations 
%
% version: Sep 2, 2018
% authors: Esa Ollila 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[N, p] = size(X);
wfun =  @(x,c) ( 1.*(x<=c) + (c*(1./x)).*(x>c) );

if nargin < 6 
    printitn = 1;
end

if nargin < 3 || isempty(c)
   c = 1.3415;  % Default: approx 95 efficiency for Gaussian errors
end

if nargin < 5 || isempty(b0)
    b0 = X \ y; %  initial start is the LSE    
end

if nargin < 4 || isempty(sig0)
    sig0 =  norm(y-X*b0)/sqrt(N-p); % initial estimate of scale    
end

csq = c^2; 
al = (1/2)*chi2cdf(csq,3)+ (csq/2)*(1-chi2cdf(csq,1)); % consistency factor for scale 
  
ITERMAX = 300;
ERRORTOL1 = 1.0e-3; % ERROR TOLERANCE FOR SCALE (joint)
ERRORTOL2 = 1.0e-3; % ERROR TOLERANCE FOR REGRESSION (joint)
ERRORTOL3 = 5.0e-4; % ERROR TOLERANCE FOR REGRESSION

Xplus = pinv(X);
con  = sqrt((N-p)*2*al);
mu = 1;
objold = 1e12; % a big number
crit1 = 1e12;
crit2 = 1e12;
mu0  = 0;
lam0 = 0;

for iter=1:ITERMAX
   
    % STEP 1: update residual  $r^{(n)}$
    r = y - X*b0;
    
    % STEP 2: update $\tau$
    tau = (norm(psihub(r/sig0,c))/con);
    
    % STEP 3: update step size for scale 
    % update only when crit 1 > 0.001
    if crit1 > 0.001 
        lam_num = norm(psihub(r/(sig0*(tau^lam0)),c))/con;
        lam = lam0 + log(lam_num)/log(tau);  
        lam = max(0.01,min(lam,1.99));
    else
        lam = lam0;
    end

    % STEP 4: update the scale 
    update1 = tau^lam;
    sig1 = sig0*update1;
   
    % STEP 5: Update $\delta$  
    psires = psihub(r/sig1,c)*sig1;  
    delta = Xplus*psires;

    % STEP 6: update the step size of regression
    % update only when crit 1 > 0.001
    if crit2 > 0.001 
        z = X*delta;
        tilde_r = r - mu0*z;
        w  = wfun(abs(tilde_r)/sig1,c);        % weights
        w(tilde_r==0) = 0.000001;
        mu2 = sum((z.^2).*w);
        mu1 = sum((z.*w).*r);
        mu =  max(0.01,min(mu1/mu2,1.99));
    else
        mu = mu0;
    end

    update2  = mu*delta;
    
    % STEP 7: update the regression vector:
    b1 = b0 + update2; 

    % STEP 8: check for convergence:    
    crit2 = norm(update2)/norm(b0);
    crit1 = abs(update1 - 1);
     
    if mod(iter,printitn)==0

      tmp1 = (y-X*b1)/sig1;
      tmp2 = psihub(tmp1,c)*sig1;     
      esteq_beta  = X'*tmp2/N;
      esteq_sigma =  sum(psihub(tmp1,c).^2)/(N-p) -2*al;

      objnew = sig1*sum(rhohub((y-X*b1)/sig1,c))/(N-p) + al*sig1;
      fprintf('%2d %.3f|%.3f\t %.9f %.9f %11.7f  %10.5f  %10.5f\n',iter,mu,lam,crit2,crit1,objnew,norm(esteq_beta),abs(esteq_sigma));
      if objnew > objold 
           fprintf('increasing objective fnc'); 
      end
    end
   
   
   if (crit2 < ERRORTOL1  && crit1 < ERRORTOL2) || crit2 < ERRORTOL3
      break 
   end   
   
   b0 = b1; 
   sig0 = sig1; 
   mu0 = mu;
   lam0 = lam;

end

if mod(iter,printitn)==0
    fprintf('\n Done! \n');
end 

if iter == ITERMAX
    fprintf('error!!! MAXiter = %d crit1 = %.7f crit2 = %.7f\n',iter,crit1,crit2);
end


