function [U1, V1, U2, V2, M] = iNeat(A, B, H, r1, r2, alpha, lambda, gamma, beta, epsi, maxiter, row1, col1, row2, col2)
% Description:
%   The algorithm "iNeat" is used for align across incomplete networks where some
%   of the network edges are missing. The algorithm uses the multiplicative
%   update to solve the nonnegative optimization problem which is
%   formulated by our two hypotheses: (1) network alignment can help
%   network completion; and (2) network completion can help alignment by
%   providing higher-quality input networks. Please refer more details to
%   the reference paper below.
%
% Input:
%   A, B: Input adjacency matrices with n1, n2 nodes.
%   H: An n2*n1 prior node similarity matrix, e.g., degree similarity.
%   r1, r2: Target ranks of the input networks A, B respectively.
%   alpha, lambda, gamma, beta, epsi: regularization parameters
%   maxiter: Maximum number of iterations.
%   row1, col1: Row/column indices of potential missing entries in A.
%   row2, col2: Row/column indices of potential missing entries in B.
%
% Output: 
%   U1, V1, U2, V2, M: Results of the low rank matrices.
%
% Reference:
%   Zhang, Si, et al. "iNEAT: Incomplete Network Alignment." 
%   Data Mining (ICDM), 2017 IEEE International Conference on. IEEE, 2017.

tol = 1e-4; 

h = norm(H, 'fro')^2;

% initializations
[U1_init, V1_init] = nnmf(A, r1); V1_init = V1_init';
[U2_init, V2_init] = nnmf(B, r2); V2_init = V2_init';

U1 = U1_init; V1 = V1_init; U2 = U2_init; V2 = V2_init;

T1 = U2'*H*U1; T2 = U2'*U2; T3 = U1'*U1;
M = max(0, inv(T2)*T1*inv(T3));

%% objective functions
P1 = projector(U1, V1, row1, col1);
P2 = projector(U2, V2, row2, col2);

T1 = U1' * U1; T2 = V1' * V1; T3 = U2' * U2; T4 = V2' * V2;
% network completion
J1 = @(U1, V1, U2, V2, P1, P2, T1, T2, T3, T4) 0.5*alpha*(norm(A, 'fro')^2 + ...
        sum(sum(T2.*T1)) - 2*sum(sum(U1.*(A*V1))))+...
        0.5*alpha*(norm(B,'fro')^2 + sum(sum(T4.*T3)) - 2*sum(sum(U2.*(B*V2)))) - ...
        0.5*alpha*(norm(P1,'fro')^2 + norm(P2,'fro')^2) + ...
        alpha*sum(sum(A.*P1)) + alpha*sum(sum(B.*P2)) + ...
        0.5*lambda*(norm(U1,'fro')^2 + norm(V1,'fro')^2 + norm(U2,'fro')^2 + norm(V2,'fro')^2);

% alignment for completion
J2 = @(U1, V1, U2, V2, M, T1, T2, T3, T4) ...
    0.5*gamma*trace(U2'*bsxfun(@times, U2*sum(V2,1)', U2)*M*(U1'*bsxfun(@times, U1*sum(V1,1)',U1)*M')) + ...
    0.5*gamma*trace(U2'*bsxfun(@times, V2*sum(U2,1)', U2)*M*(U1'*bsxfun(@times, V1*sum(U1,1)',U1)*M')) - ...
    gamma*trace(T3*(V2'*U2)*M*(U1'*V1)*T1*M');

% completion for alignment 
J3 = @(U1, V1, U2, V2, M, P1, P2) 0.5*beta*term2(A, B, U1, V1, U2, V2, M, P1, P2, row1, col1, row2, col2);

% regularization by prior information matrix H
J4 = @(U1, U2, M, H, T1, T3) 0.5*epsi*(h + trace((U2'*U2)*M*(U1'*U1)*M')-2*trace(M'*U2'*(H*U1)));

% overall objective function
J = @(U1, V1, U2, V2, M, P1, P2, T1, T2, T3, T4) J1(U1, V1, U2, V2, P1, P2, T1, T2, T3, T4) + ...
    J2(U1, V1, U2, V2, M, T1, T2, T3, T4) + J3(U1, V1, U2, V2, M, P1, P2) + J4(U1, U2, M, H, T1, T3);


func = J(U1, V1, U2, V2, M, P1, P2, T1, T2, T3, T4); 

%% alternative gradient descent
for iter = 1: maxiter
    changes = zeros(5, 1);
    tic;
    old_value = func; old_M = M; 
    old_U1 = U1; old_V1 = V1;
    old_U2 = U2; old_V2 = V2;
    
    
    % update V1
    [out1, out2] = grad_V1(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2);
    V1 = max(0, V1 .* ((out1 ./ out2) .^ (1/4)));

    changes(4) = max(max(abs((V1 - old_V1)/(1e-10 + max(max(old_V1))))));
    if sum(sum(isnan(V1))) > 0
        disp(2);
        V1 = old_V1;
        break;
    end
    P1 = projector(U1, V1, row1, col1);

    % update V2
    [out1, out2] = grad_V2(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2);

    V2 = max(0, V2 .* ((out1 ./ out2) .^ (1/4)));
    changes(5) = max(max(abs((V2 - old_V2)/(1e-10 + max(max(old_V2))))));
    if sum(sum(isnan(V2))) > 0
        disp(4);
        V2 = old_V2;
        break;
    end
    P2 = projector(U2, V2, row2, col2);
    
    
    % update M
    [out1, out2] = grad_M(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2);
    M = max(0, M .* ((out1 ./ out2) .^ (1/4)));

    changes(1) = max(max(abs((M - old_M)/(1e-10 + max(max(old_M))))));
    if sum(sum(isnan(M))) > 0
        disp(1);
        old_M = M;
        break;
    end
    
     % update U1
    [out1, out2] = grad_U1(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2);
    U1 = max(0, U1 .* ((out1 ./ out2) .^ (1/4)));
%     U1 = U1 .* ((out1 ./ out2) .^ (1/4));
    changes(2) = max(max(abs((U1 - old_U1)/(1e-10 + max(max(old_U1))))));
    if sum(sum(isnan(U1))) > 0
        disp(3);
        U1 = old_U1;
        break;
    end
    P1 = projector(U1, V1, row1, col1);
    
    % update U2
    [out1, out2] = grad_U2(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2);
    U2 = max(0, U2 .* ((out1 ./ out2) .^ (1/4)));
%     U2 = U2 .* ((out1 ./ out2) .^ (1/4));
    changes(3) = max(max(abs((U2 - old_U2)/(1e-10 + max(max(old_U2))))));
    if sum(sum(isnan(U2))) > 0
        disp(5);
        U2 = old_U2;
        break;
    end
    P2 = projector(U2, V2, row2, col2);

    maxChange = max(changes);
    
    if maxChange <= tol
        break;
    end
    
    % calculate new objective function value
    T1 = U1' * U1; T2 = V1' * V1; T3 = U2' * U2; T4 = V2' * V2;

    x1 = J1(U1, V1, U2, V2, P1, P2, T1, T2, T3, T4); 
    x2 = J2(U1, V1, U2, V2, M, T1, T2, T3, T4); 
    x3 = J3(U1, V1, U2, V2, M, P1, P2); 
    x4 = J4(U1, U2, M, H, T1, T3);
    func = x1 + x2 + x3 + x4;
    
    err = func - old_value;
    t = toc;
    fprintf('iteration %d,  maxChange = %.4f, time = %.2f\n', iter,  maxChange, t);
    if err > 0 
        U1 = old_U1; V1 = old_V1;
        U2 = old_U2; V2 = old_V2;
        M = old_M;
        break;
    end
    
    if abs(err) < tol, break; end
    
end


end




function J = term2(A, B, U1, V1, U2, V2, M, P1, P2, row1, col1, row2, col2)

x = (U1*(M'*sum(U2', 2))).^(-1); x(x == Inf) = 0;
y = (U2*(M*sum(U1', 2))).^(-1); y(y == Inf) = 0;

temp1 = M*U1'; temp2 = (U2*M)';
VV1 = temp2*B*temp2'*U1'; VV2 = temp1*A*temp1'*U2';
X = bsxfun(@times, x, bsxfun(@times, x', projector(U1, VV1', row1, col1)));
Y = bsxfun(@times, y, bsxfun(@times, y', projector(U2, VV2', row2, col2)));

J = norm(P1-X, 'fro')^2 + norm(P2-Y, 'fro')^2;

end


%% define gradients
function [out1, out2] = grad_U1(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2)

n1 = size(A,1); 
% gradient corresponding to first term, i.e., imputation only 
out11 = alpha * A * V1; 
out21 = alpha*(U1*(V1'*V1) - P1*V1) + lambda*U1;

% gradient corresponding to 'gamma' term, i.e., final consistency
temp1 = U1*M'; temp2 = U2*M; d1 = U1*sum(V1, 1)'; d2 = U2*sum(V2, 1)';
temp3 = U2'*bsxfun(@times, d2, U2); temp4 = U2'*U2; temp5 = V2'*U2;
temp6 = U1'*V1; temp7 = U1'*U1;
x1 = sum(((temp1*temp3*M).*U1), 2)*sum(V1,1);
x2 = 2*bsxfun(@times, d1, temp1)*temp3*M;
T = M'*temp4*temp5*M;
x3 = U1*(T*temp6);
x4 = V1*(temp7*T);
x5 = U1*(temp6'*T');

d11 = V1*sum(U1,1)'; d22 = V2*sum(U2,1)';
t1 = U2'*bsxfun(@times, d22, U2); 
v1 = repmat(sum((M'*t1*temp1').*U1', 1)*V1, [n1,1]);
v2 = 2*bsxfun(@times, d11, temp1)*t1*M;

out12 = gamma*(x3+x4+x5);
out22 = gamma*(0.5*(x1+x2)+0.5*(v1+v2));

% gradient corresponding to 'beta' term, i.e., cross-net completion
y = (U1 * (M' * sum(U2', 2))).^(-1); y(y == Inf) = 0; 
x = y.^2; 
yy = (U2 * (M * sum(U1', 2))).^(-1); yy(yy == Inf) = 0;
xx = yy.^2;

temp8 = U1*(temp2'*B*temp2); temp9 = projector(U1, temp8, row1, col1);
out13 = 2*beta*((y.^3).*((temp9.^2)*x))*sum(U2*M, 1) + ...
        beta*bsxfun(@times, y, bsxfun(@times, y', temp9))*V1 + ...
        beta*bsxfun(@times, y, bsxfun(@times, y', P1+P1'))*temp8;
out23 = beta*(P1 * V1 + 2 * bsxfun(@times, x, bsxfun(@times, x', temp9)) * temp8) + ...
        beta*(x .* (((P1 + P1') .* temp9) * y)) * sum(U2 * M, 1);

temp10 = A*temp1; temp11 = U2*(temp1'*A*temp1); temp12 = projector(U2, temp11, row2, col2);
out14 = beta*2*repmat(((yy'.^3).*(xx'*(temp12.^2)))*temp2, [n1, 1]) + ...
        beta*temp10*(U2'*(bsxfun(@times, yy, bsxfun(@times, yy', P2+P2')))*temp2);
out24 = beta*2*temp10*(U2'*(bsxfun(@times, xx, bsxfun(@times, xx', temp12)))*temp2) + ...
        beta*repmat((xx'.*(yy'*((P2+P2').*temp12)))*temp2, [n1, 1]);

out15 = epsi*H'*temp2; out25 = epsi*temp1*temp4*M;

out1 = out11+out12+out13+out14+out15;
out2 = out21+out22+out23+out24+out25;
out2 = out2 + eps(out1);


end

function [out1, out2] = grad_V1(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2)

n1 = size(A,1); n2 = size(B,1);
out1 = alpha*A*U1;
out2 = alpha*(V1*(U1'*U1)-P1'*U1)+lambda*V1;
% gradient corresponding to 'gamma' term, i.e., final consistency
temp1 = U1*M'; temp2 = U2*M; d2 = U2*sum(V2,1)';
temp3 = U2'*bsxfun(@times, d2, U2); 
temp4 = U1'*U1; temp5 = U2'*V2;
x1 = repmat(sum(((M'*temp3*temp1').*U1')*U1, 1), [n1, 1]);
x2 = temp1*temp5*(U2'*temp2)*temp4;

d22 = V2*sum(U2,1)';
t1 = U2'*bsxfun(@times, d22, U2);
v1 = sum((temp1*t1*M).*U1, 2)*sum(U1,1);

out1 = out1+gamma*x2;
out2 = out2+gamma*0.5*(x1+v1);

y = (U1 * (M' * sum(U2', 2))).^(-1); y(y == Inf) = 0;
temp5 = U1*(temp2'*B*temp2); temp6 = projector(U1, temp5, row1, col1);
out1 = out1+beta*bsxfun(@times, y, bsxfun(@times, y', temp6))*U1;
out2 = out2+beta*P1'*U1;


out2 = out2 + eps(out1);


end

function [out1, out2] = grad_U2(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2)

n2 = size(B,1);

out1 = alpha*B*V2;
out2 = alpha*(U2*(V2'*V2)-P2*V2)+lambda*U2;
% gradient corresponding to 'gamma' term, i.e., final consistency
temp1 = U1*M'; temp2 = U2*M; d1 = U1*sum(V1, 1)'; d2 = U2*sum(V2, 1)';
temp3 = U1'*bsxfun(@times, d1, U1); temp4 = U1'*U1; temp5 = V1'*U1;
temp6 = U2'*V2; temp7 = U2'*U2;
x1 = sum(((temp2*temp3*M').*U2), 2)*sum(V2,1);
x2 = 2*bsxfun(@times, d2, temp2)*temp3*M';
x3 = temp2*(temp4*temp5*M'*temp6);
x4 = V2*(temp7*M*temp4*temp5*M');
x5 = U2*(temp6'*M*temp5'*temp4*M');

d11 = V1*sum(U1,1)'; d22 = V2*sum(U2,1)';
t1 = U1'*bsxfun(@times, d11, U1);
v1 = repmat(sum((M*t1*temp2').*U2',1)*V2, [n2,1]);
v2 = 2*bsxfun(@times, d22, temp2)*t1*M';

out1 = out1+gamma*(x3+x4+x5);
out2 = out2+0.5*gamma*(x1+x2+v1+v2);

y = (U2*(M*sum(U1', 2))).^(-1); y(y == Inf) = 0;
x = y.^2;
temp8 = U2*(temp1'*A*temp1); temp9 = projector(U2, temp8, row2, col2);
out1 = out1+2*beta*(y.^3.*((temp9.^2)*x))*sum(U1*M',1)+...
       beta*bsxfun(@times, y, bsxfun(@times, y', temp9))*V2 + ...
       beta*bsxfun(@times, y, bsxfun(@times, y', P2+P2'))*temp8;
out2 = out2+beta*(P2*V2+2*bsxfun(@times, x, bsxfun(@times, x', temp9))*temp8) + ...
       beta*(x.*(((P2+P2').*temp9)*y))*sum(U1*M',1);

yy = (U1*(M'*sum(U2',2))).^(-1); yy(yy == Inf) = 0;
xx = yy.^2;
temp10 = B*temp2; temp11 = U1*(temp2'*B*temp2); temp12 = projector(U1, temp11, row1, col1);
out1 = out1+beta*2*repmat((yy'.^3.*(xx'*(temp12.^2)))*temp1, [n2, 1]) + ...
       beta*temp10*(U1'*bsxfun(@times, yy, bsxfun(@times, yy', P1+P1'))*temp1);
out2 = out2+beta*2*temp10*(U1'*bsxfun(@times, xx, bsxfun(@times, xx', temp12))*temp1)+...
       beta*repmat((xx'.*(yy'*((P1+P1').*temp12)))*temp1, [n2, 1]);
   
out1 = out1 + epsi*H*temp1;
out2 = out2 + epsi*temp2*temp4*M';
out2 = out2 + eps(out1);

end
  
function [out1, out2] = grad_V2(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2)

n1 = size(A,1); n2 = size(B,1);
out1 = alpha*B*U2; 
out2 = alpha*(V2*(U2'*U2)-P2'*U2)+lambda*V2;
% gradient corresponding to 'gamma' term, i.e., final consistency
temp1 = U1*M'; temp2 = U2*M; d1 = U1*sum(V1,1)';
temp3 = U1'*bsxfun(@times, d1, U1); temp4 = U1'*U1; temp5 = U1'*V1;
temp6 = U2'*U2;
x1 = repmat(sum((M*temp3*temp2').*U2', 1)*U2, [n2,1]);
x2 = temp2*(temp5*temp4*M'*temp6);

d11 = V1*sum(U1,1)';
t1 = U1'*bsxfun(@times, d11, U1);
v1 = sum((temp2*t1*M').*U2, 2)*sum(U2,1);

out1 = out1+gamma*x2; 
out2 = out2+0.5*gamma*(x1+v1);

y = (U2*(M*sum(U1',2))).^(-1); y(y == Inf) = 0;
temp7 = U2*(temp1'*A*temp1); temp8 = projector(U2, temp7, row2, col2);
out1 = out1+beta*bsxfun(@times, y, bsxfun(@times, y', temp8))*U2;
out2 = out2+beta*P2'*U2;
out2 = out2+eps(out1);

end


function [out1, out2] = grad_M(A, B, U1, V1, U2, V2, M, H, P1, P2, alpha, lambda, gamma, beta, epsi, row1, col1, row2, col2)
n1 = size(A,1); n2 = size(B,1);
% gradient corresponding to 'gamma' term, i.e., final consistency
temp1 = U1*M'; temp2 = U2*M; 
d11 = U1*sum(V1,1)'; d12 = V1*sum(U1,1)'; d21 = U2*sum(V2,1)'; d22 = V2*sum(U2,1)';
% temp3 = U1'*bsxfun(@times, d1, U1); temp4 = U2'*bsxfun(@times, d2, U2); 
temp31 = U1'*bsxfun(@times, d11, U1); temp41 = U2'*bsxfun(@times, d21, U2); 
temp32 = U1'*bsxfun(@times, d12, U1); temp42 = U2'*bsxfun(@times, d22, U2); 
temp5 = U1'*U1; temp6 = U1'*V1; temp7 = U2'*U2; temp8 = U2'*V2;

x1 = temp41*M*temp31+temp42*M*temp32;
x2 = temp7*temp8'*M*temp6*temp5;
x3 = temp8*temp7*M*temp5*temp6';

out1 = gamma*(x2+x3);
out2 = gamma*x1;

y = (U1*(M'*sum(U2', 2))).^(-1); y(y == Inf) = 0;
x = y.^2;
temp9 = U2'*B*U2*temp1'; temp10 = U1*(temp2'*B*temp2); temp11 = projector(U1, temp10, row1, col1);
out1 = out1+2*beta*sum(U2', 2)*(y'.^3.*((x'*(temp11.^2))))*U1 + ...
       beta*temp9*bsxfun(@times, y, bsxfun(@times, y', P1+P1'))*U1;
out2 = out2+2*beta*temp9*bsxfun(@times, x, bsxfun(@times, x', temp11))*U1 + ...
       beta*sum(U2', 2)*(x'.*(y'*((P1+P1').*temp11)))*U1;

yy = (U2*(M*sum(U1', 2))).^(-1); yy(yy == Inf) = 0;
xx = yy.^2;
temp12 = temp2*(U1'*A*U1); temp13 = temp12*M'; temp14 =projector(U2,temp13,row2,col2);
out1 = out1+beta*2*U2'*(yy.^3.*((temp14.^2)*xx))*sum(U1,1) + ...
       beta*U2'*bsxfun(@times, yy, bsxfun(@times, yy', P2+P2'))*temp12;
out2 = out2+2*beta*U2'*bsxfun(@times, xx, bsxfun(@times, xx', temp14))*temp12 + ...
       beta*U2'*(xx.*(((P2+P2').*temp14)*yy))*sum(U1,1);
   
out1 = epsi*U2'*(H*U1); out2 = epsi*temp7*M*temp5;
out2 = out2+eps(out1);
   
end

function P = projector(U, V, row, col)

    n = size(U, 1);
    v = zeros(length(row),1);
    for i = 1:length(row)
        v(i) = U(row(i), :)*V(col(i), :)';
    end
    P = sparse(row, col, v, n, n);

end


