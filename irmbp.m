function [X, itr_vec, obj, obj_out, obj_in] = irmbp(Y,D,X_ini,prm)

M = size(D,1);
N = size(D,2);
T = size(Y,2);

%Parameters
p = prm.p;
q = prm.q;
max_itr = prm.max_itr;
max_itr_l1q = prm.max_itr_l1q;
thr_x = prm.thr_x;
thr_x_l1q = prm.thr_x_l1q;
lambda = prm.lambda;

%Initialize
obj = zeros(max_itr,max_itr_l1q);
obj_out = zeros(max_itr,1);
obj_in = zeros(max_itr,max_itr_l1q);
itr_vec = zeros(max_itr,1);
X  = X_ini;
z_n = ones(N,1);

%Iteration index
itr = 1;
obj_out(itr) = (1/2)*norm(Y-D*X,'fro')^2+(lambda/p)*sum(sum(abs(X).^q,2).^(p/q));

while(1)
    Xold = X;

    %%%%% Solve l1q-norm minimization %%%%%
    
    %Initialize for l1q-norm minimization
    X_l1q = X;
    
    itr_l1q = 1;
    obj(itr,itr_l1q) = (1/2)*norm(Y-D*X_l1q,'fro')^2+(lambda/p)*sum(sum(abs(X_l1q).^q,2).^(p/q));
    obj_in(itr,itr_l1q) = (1/2)*norm(Y-D*X_l1q,'fro')^2+lambda*sum(z_n.*sum(abs(X_l1q).^q,2).^(1/q));
    
    while (1)
        Xold_l1q = X_l1q;
        

        eta_n = sum(abs(X_l1q).^q,2);
        eta_nt = abs(X_l1q).^2;

        for t=1:T
            W = diag(sqrt((1./z_n).*eta_n.^(1-1/q).*eta_nt(:,t).^(1-q/2)));
            A = D*W;
            X_l1q(:,t) = W*A'/(A*A'+lambda*eye(M))*Y(:,t);
        end
        
        dX_l1q =  norm(X_l1q - Xold_l1q,'fro')^2;

%         fprintf('[L1q] itr: %d, obj: %f, dX: %f\n',itr_l1q, obj_l1q(itr), dX_l1q);
        
        itr_l1q = itr_l1q + 1; %Increment iteration index
        obj(itr,itr_l1q) = (1/2)*norm(Y-D*X_l1q,'fro')^2+(lambda/p)*sum(sum(abs(X_l1q).^q,2).^(p/q));
        obj_in(itr,itr_l1q) = (1/2)*norm(Y-D*X_l1q,'fro')^2+lambda*sum(z_n.*sum(abs(X_l1q).^q,2).^(1/q));
        
        %Stopping condition
        if itr_l1q>max_itr_l1q; break; end;
        if dX_l1q < thr_x_l1q; break; end;
    end
    
    itr_vec(itr) = itr_l1q;
    X = X_l1q;
    
    %%%%% Update weights %%%%%
    z_n = 1./(((sum(abs(X).^q,2)).^(1/q)).^(1-p));
    
    itr = itr + 1;
    obj_out(itr) = (1/2)*norm(Y-D*X,'fro')^2+(lambda/p)*sum(sum(abs(X).^q,2).^(p/q));
    dX =  norm(X - Xold,'fro')^2;

%     fprintf('itr: %d, obj: %f, dX: %f\n',itr, obj(itr), dX);

    %Stopping condition
    if itr>max_itr; break; end;
    if dX < thr_x; break; end;
    
end

obj = obj(1:itr-1,:);
obj_out = obj_out(1:itr);
obj_in = obj_in(1:itr-1,:);
itr_vec = itr_vec(1:itr-1);

end