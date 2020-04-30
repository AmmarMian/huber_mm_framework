clear;
%%
% simulation parameters
N = 2^7  % number of measurements 
p = 2^8  % dimension of measurement matrix
K = 2.^3 % number of nonzeros in a signal vector 
sig = sqrt(1/10); % 10 dB 
SNR = 20*log10(1/sig)
%v = 5;  % d.o.f
%const_nu = 0.726687; % med(|e|)/const=1 for t_5
v = 3;
const_nu = 0.764892; % med(|e|)/const=1 for t_5
%%

NRSIM = 1000; 
MSE1 = zeros(NRSIM,1);
MSE2 = zeros(NRSIM,1);
REC1 = zeros(NRSIM,1); 
REC2 = zeros(NRSIM,1); 
time1 = 0;
time2 = 0;
it1 = 0;
it2 = 0;
fail1 = 0;
fail2 = 0;
rng('default');

%%
    
for iter = 1:NRSIM
    
    %% generate the data
  
    % signal vector with k-zeros
    bnz = zeros(p,1);
    loc = sort(randsample(p,K));  % random locations of nonzeros
    beta = zeros(p,1);
    beta(loc) =  1; 

    % measurement matrix with unit norm columns
    X= randn(N,p);
    len = sqrt(sum(X.*X));  % norms 
    X = X.*repmat(1./len,N,1);    

    % generate the noise e from t_5-distribution and data y = X*beta + e
    e =(sig/const_nu)*trnd(v,N,1); % t_5(0,sig) noise with sig = MEdian(|e|)
    y = X*beta + e;

    % HUB-NIHT
    
    tStart = tic;
    [b1,sig1,supp1,iter1,failure1] = hubniht(y,X,K);
    length(setdiff(loc,supp1))
    time2 = time2 + toc(tStart);
    it1 = it1+ iter1;
    fail1 = fail1 + failure1;
    if  isempty(setdiff(supp1,loc))
        REC1(iter) = 1;
    end
    MSE1(iter) = norm(b1-beta)^2;
    
    %% NIHT
    tStart = tic;
    [b2, supp2, failure2,iter2] = sniht(y,X,K);
    time2 = time2 + toc(tStart);
    length(setdiff(loc,supp2))
    it2 = it2+ iter2;
    fail2 = fail2 + failure2;
    if  isempty(setdiff(supp1,loc))
        REC1(iter) = 1;
    end
    MSE1(iter) = norm(b2-beta)^2;

    %%
end




