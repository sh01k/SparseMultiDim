function [arg1_opt,arg2_opt,val1_mat,val2_mat]=prmgrid_sumn(arg1_vec,arg2_vec,Y,D,X_ini,Xtrue,Xrow_ind,prm)

val1_mat = zeros(length(arg1_vec),length(arg2_vec));
val2_mat = zeros(length(arg1_vec),length(arg2_vec));

fmsr_thr = min(sum(abs(Xtrue(Xrow_ind,:)).^2,2))*1e-2;

for ii=1:length(arg1_vec)
    arg1 = arg1_vec(ii);
    prm.lambda = arg1;
    for jj=1:length(arg2_vec)
        arg2 = arg2_vec(jj);
        prm.mu = arg2;
        
        X = sumnorm(Y,D,X_ini,prm);
        val1_mat(ii,jj) = 10*log10(sum(abs(Xtrue).^2)/sum(abs(X-Xtrue).^2));
        
        Xrow = sum(abs(X).^2,2);
        ind = find(Xrow>fmsr_thr);
        val2_mat(ii,jj) = 2*numel(intersect(Xrow_ind,ind))/(numel(ind)+numel(Xrow_ind));
    end
end

[~,max_idx] = max(val1_mat(:));
[idx1,idx2] = ind2sub(size(val1_mat),max_idx);
arg1_opt = arg1_vec(idx1);
arg2_opt = arg1_vec(idx2);

fprintf('Optimal argument: %f, %f\n',arg1_opt,arg2_opt);

end