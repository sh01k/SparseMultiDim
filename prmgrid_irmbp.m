function [arg_opt,val1_vec,val2_vec]=prmgrid_irmbp(arg_vec,Y,D,X_ini,Xtrue,Xrow_ind,prm)

val1_vec = zeros(length(arg_vec),1);
val2_vec = zeros(length(arg_vec),1);

fmsr_thr = min(sum(abs(Xtrue(Xrow_ind,:)).^2,2))*1e-2;

for ii=1:length(arg_vec)
    arg = arg_vec(ii);
    prm.lambda = arg;
    X = irmbp(Y,D,X_ini,prm);
    val1_vec(ii) = 10*log10(sum(abs(Xtrue).^2)/sum(abs(X-Xtrue).^2));
    Xrow = sum(abs(X).^2,2);
    ind = find(Xrow>fmsr_thr);
    val2_vec(ii) = 2*numel(intersect(Xrow_ind,ind))/(numel(ind)+numel(Xrow_ind));
end

[~,max_idx] = max(val1_vec);
arg_opt = arg_vec(max_idx);

fprintf('Optimal argument: %f\n',arg_opt);

end