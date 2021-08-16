function [KP, KQ] = affinity(A1, A2, N1, N2, E1, E2)


n1 = size(A1, 1); n2 = size(A2, 1);

% If no node attributes input, then initialize as a vector of 1
% so that all nodes are treated to have the save attributes which 
% is equivalent to no given node attribute.
if isempty(N1) && isempty(N2)
    N1 = ones(n1, 1);
    N2 = ones(n2, 1);
end

% If no edge attributes are input, i.e., E1 and E2 are empty, then
% initialize as a cell with 1 element, which is same as adjacency matrix
% but the entries that are nonzero in adjacency matrix are equal to 1 so 
% that all edges are treated as with the same edge attributes. This is 
% equivalent to no given edge attributes.
if isempty(E1) && isempty(E2)
    E1 = cell(1,1); E2 = cell(1,1);
    E1{1} = A1; E2{1} = A2;
    E1{1}(A1 > 0) = 1; E2{1}(A2 > 0) = 1;
end
    
L = size(E1, 2);

% Normalize edge feature vectors 

edge_idx1 = find(triu(A1)); m1 = length(edge_idx1);
edge_idx2 = find(triu(A2)); m2 = length(edge_idx2);
edge_attr1 = zeros(m1, L); edge_attr2 = zeros(m2, L);

for i = 1: L
    edge_attr1(:, i) = E1{i}(edge_idx1);
    edge_attr2(:, i) = E2{i}(edge_idx2);
end

KQ = zeros(length(edge_attr1), length(edge_attr2));
for i = 1: length(edge_attr1)
    KQ(i, :) = 1 - abs(edge_attr1(i)-edge_attr2')./max(edge_attr1(i), edge_attr2');
end


% Normalize node feature vectors
K1 = sum(N1.^2, 2).^(-0.5); K1(K1 == Inf) = 0;
K2 = sum(N2.^2, 2).^(-0.5); K2(K2 == Inf) = 0;
% isequal(bsxfun(@times, K1, N1), normr(N1))
N1 = bsxfun(@times, K1, N1); % normalize the node attribute for A1
N2 = bsxfun(@times, K2, N2); % normalize the node attribute for A2

KP = N1 * N2';


end