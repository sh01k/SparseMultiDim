function [X, itr, obj] = sumnorm(Y,D,X_ini,prm)

M = size(D,1);
% N = size(D,2);
T = size(Y,2);

%Parameters
p = prm.p;
q = prm.q;
max_itr = prm.max_itr; %Maximum number of iterations
thr_x = prm.thr_x;
lambda = prm.lambda; %Reguralization parameter
mu = prm.mu; %Reguralization parameter

%Initialize
X = X_ini;
obj = zeros(max_itr,1);

%Iteration index
itr = 1;
obj(itr) = (1/2)*norm(Y-D*X,'fro')^2+(lambda/p)*sum(sum(abs(X).^q,2).^(p/q));

while (1)
    Xold = X;
    
    eta_n = sum(abs(X).^2,2);
    eta_nt = abs(X).^2;
    
    for t=1:T
        W = diag((eta_n.^(p/2-1)+mu*(q/p)*eta_nt(:,t).^(q/2-1)).^(-1/2));
        A = D*W;
        X(:,t) = W*A'/(A*A'+lambda*eye(M))*Y(:,t);
    end
    
%     fprintf('itr: %d, obj: %f, dX: %f\n',itr, obj(itr), dX);
    
    itr = itr + 1; %Increment iteration index
    obj(itr) = (1/2)*norm(Y-D*X,'fro')^2+(lambda/p)*sum(sum(abs(X).^q,2).^(p/q));
    dX =  norm(X - Xold,'fro')^2;
    
    %Stopping condition
    if itr>max_itr; break; end;
    if dX < thr_x; break; end;
    
end

obj = obj(1:itr);

end