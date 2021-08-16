clear; clc;
ratio = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25];

acc12_ineat = zeros(length(ratio), 1);

for i = 1: length(ratio)
    
    filename = sprintf('Gordian-v2-%d%%.mat', ratio(i)*100);
    load(filename);
    N1 = full(N1); N2 = full(N2); N3 = full(N3);
    H12 = normr(N1)*normr(N2)';

    % run iNeat

    [U1, V1, U2, V2, M] = iNeat(C1, C2, H12'*10, 200, 200, 1, 1, 1, 1, 1, 20, r1, c1, r2, c2);
    U12.U1{i} = U1; V12.V1{i} = V1; U12.U2{i} = U2;
    V12.V2{i} = V2; M12.M{i} = M;
    
    S12 = 0.5*(U2*M*U1' + H12'); X12 = greedy_match(S12); 
    [row, col] = find(X12); 
    acc12_ineat(i) = length(intersect([col row], gnd, 'rows'))/length(gnd);

    if ~exist('Gordian-v2_res.mat')
        save('Gordian-v2_res.mat','acc12_ineat','U12','V12','M12');
    else
        save('Gordian-v2_res.mat','acc12_ineat','U12','V12','M12','-append')
    end

end