addpath(genpath('fgm-master'));
prSet(1);
%% algorithm parameter
[pars, algs] = gmPar(2);

ratio = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25];
acc_rrwr_incomplete = zeros(length(ratio), 1);

for i = 1: length(ratio)
    
    fprintf('running ratio=%.2f.\n', ratio(i));
    filename1 = sprintf('Gordian-v2-%d%%.mat', ratio(i)*100);
    load(filename1);
   
    fprintf('start constructing Kronecker product.\n');

    B1 = C1; B2 = C2; B1(B1 ~= 0) = 1; B2(B2 ~= 0) = 1;
    K = kron(B1, B2); [p, q] = size(C2); n1= size(C1, 1);

    H = H12; h = H(:); 
    D = spdiags(h(:), 0, length(h), length(h));
    K = K + D; 

    Ct = ones(n1, p);

    fprintf('finished constructing Kronecker matrix.\n');
    asgRrwm = gm(D, Ct, [], pars{7}{:});

    [row, col] = find(asgRrwm.X);
    acc_rrwr_incomplete(i) = length(intersect([col, row], gnd, 'rows'))/length(gnd);
    if ~exist('Gordian-v2_res.mat')
        save('Gordian-v2_res.mat','acc_rrwr_incomplete');
    else
        save('Gordian-v2_res.mat','acc_rrwr_incomplete','-append')
    end
    
end
