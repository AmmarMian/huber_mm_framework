clearvars;
n = 500;
p = 250;
SNRdb = 20;
MCiter = 2000;
errprob  = [0 0.01:0.01:0.10];
Nerr = length(errprob);

SD = zeros(MCiter,Nerr); 
errLS = zeros(1,Nerr);
errH = zeros(1,Nerr); 
Hsig = zeros(MCiter,Nerr); 
siglist = zeros(MCiter,1);
iter  = zeros(1,Nerr);
time  = zeros(2,Nerr);
%%

for kk = 1:length(errprob)
    
    err_prob = errprob(kk);
    rng('default');  % set random seeds  
    %%
    for ii = 1:MCiter 

        %% Generate the data set 
        absb = unifrnd(0,10,p,1);
        b = absb.*sign(unifrnd(-1,1,p,1));

        X  = randn(n,p); % design matrix 
        e0 = randn(n,1);  
        con = (norm(X*b)/norm(e0))^2; 
        sig = sqrt(con*10^(-SNRdb/10));
        siglist(ii,1)=sig;
        
        % 10*log10(norm(X*b)^2/(sig^2*norm(e0)^2)) % should be SNRdb 
        y = X*b + sig*e0;     
    
        % flip the orientation! 
        if err_prob > 0
            indx = binornd(1,err_prob,n,1);
            %y(logical(indx)) = exp(1i*pi/2)*y((logical(indx)));    
            y(logical(indx)) = -y((logical(indx)));    
        end
  
        %% LS-estimates of regression and scale:
        tstart = tic; 
        LSE = X \ y; 
        s0 = norm(y-X*LSE)/sqrt(n-p); 
        time(1,kk)=time(1,kk) + toc(tstart);
        
        errLS(1,kk) = errLS(1,kk) + norm(LSE - b)^2/norm(b)^2;
        SD(ii,kk) = s0;
        %errLS(2,kk) = errLS(2,kk) +  log10(s0/sig)^2 ;
        
        %% Huber's estimate of regression 
        
        tstart = tic;
        [best2,sigest2,it] = hubreg(y,X);
        time(2,kk) = time(2,kk) + toc(tstart);
        
        iter(kk)   = iter(kk)+it;
        errH(1,kk) = errH(1,kk) + norm(best2 - b)^2/norm(b)^2;  
        Hsig(ii,kk) = sigest2;

        %errH(2,kk) = errH(2,kk) + log10(sigest2/sig)^2;
        %%  
               
        if mod(ii,200)==0, fprintf('. '); end
        %%         
    end
    
    fprintf(' Done with err_prob =%.2f\n',err_prob);
end

[repmat(mean(siglist),1,Nerr);mean(SD);mean(Hsig)]
errSD = log10(SD./repmat(siglist,1,Nerr)).^2;
errHsig = log10(Hsig./repmat(siglist,1,Nerr)).^2;


%% FIGURE 1
% plot of error probability vs MSE
h=figure(1); clf;
plot(errprob,errLS/MCiter,'*-','LineWidth',1.5,'MarkerSize',12)
set(gca,'XTick',[0:0.02:0.1],'FontSize',16)    
set(gca,'YTick',[0 0.05:0.05:0.4],'FontSize',16) 
axis tight; 
hold on;
plot(errprob,errH/MCiter,'o-','LineWidth',1.5,'MarkerSize',12)
xlabel('$\epsilon$','Interpreter','Latex')
ylabel('$\| \hat{\beta} - \beta \|^2_2/\| \beta \|^2_2$', 'Interpreter','Latex')
h_legend=legend('LSE','HUBReg $q=.95$'); 
set(h_legend,'FontSize',20,'Interpreter','Latex','Location','Best');
ax = axis;
axis([ax(1:2) -0.01 ax(4)])
grid on
set(gca,'FontSize',18,'FontName','TimesNewRoman')

%%  FIGURE2 (zoom in)
h=figure(2); clf;
plot(errprob(1:9),errLS(1,1:9)/MCiter,'*-','LineWidth',1.5,'MarkerSize',12)
set(gca,'XTick',0:0.02:0.08)    
set(gca,'YTick',[0.01:0.004:0.03]) 
axis tight; 
hold on;
axis([0 0.08 0.01 0.03])
plot(errprob(1:9),errH(1,1:9)/MCiter,'o-','LineWidth',1.5,'MarkerSize',12)
xlabel('\epsilon')
ylabel('$\| \hat{\beta} - \beta \|^2_2/\| \beta \|^2_2$','Interpreter','Latex')
h_legend=legend('LSE','HUBReg','Location','Best'); 
set(h_legend,'FontSize',20,'Interpreter','Latex');
grid on
set(gca,'FontSize',18,'FontName','TimesNewRoman')

%%
h=figure(3); clf;
plot(errprob,mean(errSD),'*-','LineWidth',1.5,'MarkerSize',12)
set(gca,'XTick',0:0.02:0.1,'FontSize',16)    
%set(gca,'YTick',[0 0.05:0.1:0.4],'FontSize',16) 
hold on;
plot(errprob,mean(errHsig),'o-','LineWidth',1.5,'MarkerSize',12)
xlabel('$\epsilon$','Interpreter','Latex')
ylabel('$\log_{10}^2(\hat \sigma/\sigma)$','Interpreter','Latex')
h_legend=legend('Standard deviation','HUB-Reg $\hat \sigma$ ($q=.95$)','Location','Best'); 
set(h_legend,'FontSize',20,'Interpreter','Latex');
axis tight 
ax = axis;
axis([ax(1:2) 0 ax(4)])
grid on
set(gca,'FontSize',18,'FontName','TimesNewRoman')

%%
h=figure(4); clf;
bar(time/MCiter)
set(gca,'XTickLabel',{'.001','.01','.02','.03','.04','.05','.06','.07','.08','.09','.10'},'FontSize',14)
xlabel('\epsilon')
ylabel('Time in seconds', 'FontSize',20);
%ax = axis;
%axis([0.5 11.5 0 0.9])
h_legend=legend('LSE','HUB-Reg $q=.95$');
set(h_legend,'FontSize',20,'Interpreter','Latex','Location','NorthWest'); 
set(gca,'FontSize',18,'FontName','TimesNewRoman')
%%

