
ratio = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25];

acc_finalP_incomplete = zeros(length(ratio), 1);
for i = 1: length(ratio)
    fprintf('running ratio=%.2f.\n', ratio(i));
    
    filename1 = sprintf('Gordian-v2-%d%%.mat', ratio(i)*100);
    load(filename1);
    fprintf('finish loading.\n');
    N1 = full(N1); N2 = full(N2); N3 = full(N3);
    H12 = normr(N1)*normr(N2)';
    fprintf('start FINAL_P.\n');
    S12_finalP = FINAL_P(C1, C2, H12', 0.5, 100, 1e-4);
    X12 = greedy_match(S12_finalP);
    [row, col] = find(X12);
    
    acc_finalP_incomplete(i) = length(intersect([col, row], gnd, 'rows'))/length(gnd);
    if ~exist('Gordian-v2_res.mat')
        save('Gordian-v2_res.mat','acc_finalP_incomplete');
    else
        save('Gordian-v2_res.mat','acc_finalP_incomplete','-append')
    end
end
