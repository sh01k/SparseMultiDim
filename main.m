clear variables;
close all;

set(0,'defaultAxesFontSize',16);
set(0,'defaultAxesFontName','Times');
set(0,'defaultTextFontSize',16);
set(0,'defaultTextFontName','Times');

%Graphic parameters
WindowLocation;

%% Parameters

M= 16;
N = 64;
T = 50;

lambda_vec = 10.^linspace(-2,2,50);
mu_vec = 10.^linspace(-2,2,50);

flg_prm_search = 0;

%% Generate dictionary

%Dictionary matrix
D = randn(M,N) + 1i*randn(M,N);
D = D*diag(1./sqrt(sum(abs(D).^2,1)));

%% Generate solution and observation vectors

%Number of activations
K = 4;

Xrow_ind = randperm(N,K);

%Activated sequence w/ Gamma distribution
kurt = 1/0.05;
shape_gam = 6/kurt;
scale_gam = 1/shape_gam;
Xrow_val = gamrnd(shape_gam,scale_gam,K,T).*exp(1i*2*pi*rand(K,T));

fig(1)=figure(1);
plot(1:T,real(Xrow_val(1,:)),1:T,abs(Xrow_val(1,:)));

Xtrue = zeros(N,T);
Xtrue(Xrow_ind,:) = Xrow_val;

Xrow_true = sum(abs(Xtrue).^2,2);

%Noise
SNR = 10;
X_pw = sum(sum(abs(Xtrue).^2))/T/M;
E_pw = X_pw*10^(-SNR/10);
E = sqrt(E_pw/2)*(randn(M,T)+1i*randn(M,T));

%Observation 
Y = D*Xtrue + E;

%% MM for Mixed-Norm Penalty

fprintf('---------- MM for Mixed-Norm Penalty ----------\n');

mixn_prm.p = 0.4;
mixn_prm.q = 1.0;
mixn_prm.max_itr = 300;
mixn_prm.thr_x = 1e-6;

%Initial value
X_mixn_ini = ones(N,T);

%Regularization paramter
if flg_prm_search == 1
    fprintf('Parameter search...\n');
    [mixn_prm.lambda,sdrd_reg_mixn,fmsr_reg_mixn] = prmgrid_mixn(lambda_vec,Y,D,X_mixn_ini,Xtrue,Xrow_ind,mixn_prm);
else
    mixn_prm.lambda = 2.81;
end

[X_mixn,itr_mixn,obj_mixn] = mixnorm(Y,D,X_mixn_ini,mixn_prm);

fprintf('Number of iterations: %d\n',itr_mixn);

Xrow_mixn = sum(abs(X_mixn).^2,2);

fig(11) = figure(11);
set(fig(11),'Position',wloc4_l1(1,:));
plot(1:N,Xrow_mixn,'-o',1:N,Xrow_true,'--x');
xlabel('Row index'); 

fig(12) = figure(12);
set(fig(12),'Position',wloc4_l2(1,:));
plot(20*log10(obj_mixn));
xlabel('Iteration'); ylabel('Objective function [dB]');

%% IrM-BP

fprintf('---------- IrM-BP ----------\n');

irbp_prm.p = 0.4;
irbp_prm.q = 1.0;
irbp_prm.max_itr = 5;
irbp_prm.max_itr_l1q = 300;
irbp_prm.thr_x = 1e-6;
irbp_prm.thr_x_l1q = 1e-6;

%Initial value
X_irbp_ini = ones(N,T);

%Regularization parameter
if flg_prm_search == 1
    fprintf('Parameter search...\n');
    [irbp_prm.lambda,sdrd_reg_irbp,fmsr_reg_irbp] = prmgrid_irmbp(lambda_vec,Y,D,X_irbp_ini,Xtrue,Xrow_ind,irbp_prm);
else
    irbp_prm.lambda = 2.81;
end

[X_irbp,itr_irbp,obj_irbp,obj_irbp_out,obj_irbp_in] = irmbp(Y,D,X_irbp_ini,irbp_prm);

fprintf('Number of iterations: %d\n',sum(itr_irbp));

obj_irbp_vec = zeros(sum(itr_irbp),1);
for ii=1:length(itr_irbp)
    obj_irbp_vec(sum(itr_irbp(1:ii-1))+1:sum(itr_irbp(1:ii))) = obj_irbp(ii,1:itr_irbp(ii));
end

Xrow_irbp = sum(abs(X_irbp).^2,2);

fig(21) = figure(21);
set(fig(21),'Position',wloc4_l1(2,:));
plot(1:N,Xrow_irbp,'-o',1:N,Xrow_true,'--x');
xlabel('Row index'); 

fig(22) = figure(22);
set(fig(22),'Position',wloc4_l2(2,:));
plot(20*log10(obj_irbp_vec));
xlabel('Iteration'); ylabel('Objective function [dB]');

%% MM for Lp2+Lq penalty

fprintf('---------- MM for Lp2+Lq penalty ----------\n');

sumn_prm.p = 1.0;
sumn_prm.q = 1.0;
sumn_prm.max_itr = 300;
sumn_prm.thr_x = 1e-6;

%Initial value
X_sumn_ini = ones(N,T);

%Regularization paramter
if flg_prm_search == 1
    fprintf('Parameter search...\n');
    [sumn_prm.lambda,sumn_prm.mu,sdrd_reg_sumn,fmsr_reg_sumn] = prmgrid_sumn(lambda_vec,mu_vec,Y,D,X_sumn_ini,Xtrue,Xrow_ind,sumn_prm);
else
    sumn_prm.lambda = 0.75;
    sumn_prm.mu = 0.36;
end

[X_sumn,itr_sumn,obj_sumn] = mixnorm(Y,D,X_sumn_ini,sumn_prm);

fprintf('Number of iterations: %d\n',sum(itr_sumn));

Xrow_sumn = sum(abs(X_sumn).^2,2);

fig(31) = figure(31);
set(fig(31),'Position',wloc4_l1(3,:));
plot(1:N,Xrow_sumn,'-o',1:N,Xrow_true,'--x');
xlabel('Row index'); 

fig(32) = figure(32);
set(fig(32),'Position',wloc4_l2(3,:));
plot(20*log10(obj_sumn));
xlabel('Iteration'); ylabel('Objective function [dB]');

%% Evaluation

%SDRD
sdrd_mixn = 10*log10(sum(sum(abs(Xtrue).^2))/sum(sum(abs(X_mixn - Xtrue).^2)));
sdrd_irbp = 10*log10(sum(sum(abs(Xtrue).^2))/sum(sum(abs(X_irbp - Xtrue).^2)));
sdrd_sumn = 10*log10(sum(sum(abs(Xtrue).^2))/sum(sum(abs(X_sumn - Xtrue).^2)));

%F-measure
fmsr_thr = min(sum(abs(Xtrue(Xrow_ind,:)).^2,2))*1e-2;

ind_mixn=find(Xrow_mixn>fmsr_thr);
fmsr_mixn = 2*numel(intersect(Xrow_ind,ind_mixn))/(numel(ind_mixn)+numel(Xrow_ind));

ind_irbp=find(Xrow_irbp>fmsr_thr);
fmsr_irbp = 2*numel(intersect(Xrow_ind,ind_irbp))/(numel(ind_irbp)+numel(Xrow_ind));

ind_sumn=find(Xrow_sumn>fmsr_thr);
fmsr_sumn = 2*numel(intersect(Xrow_ind,ind_sumn))/(numel(ind_sumn)+numel(Xrow_ind));

fprintf('===== Evaluation =====\n');
fprintf('[SDRD] Mixed-norm: %f, IrM-BP: %f, Sum-norm: %f\n', sdrd_mixn, sdrd_irbp, sdrd_sumn);
fprintf('[Fmsr] Mixed-norm: %f, IrM-BP: %f, Sum-norm: %f\n', fmsr_mixn, fmsr_irbp, fmsr_sumn);

%% Terminate
