function [mu_m, mu_2] = find_eig(Lg)
%FIND_EIG Compute extreme eigenvalues of a graph Laplacian.
%   [MU_M, MU_2] = FIND_EIG(LG) returns:
%       MU_M : largest eigenvalue of the Laplacian LG
%       MU_2 : algebraic connectivity (second-smallest eigenvalue of LG)
%
%   Input:
%     LG : Laplacian matrix of the communication graph (m-by-m).
%
%   Output:
%     MU_M : max eigenvalue of LG
%     MU_2 : second-smallest eigenvalue of LG (connectivity measure)
%
%   Notes:
%     • The smallest eigenvalue of LG is always zero for a connected graph.
%     • Used in DKF consensus/stability conditions.

    mu = eigs(Lg, size(Lg, 1));  % eigenvalues (possibly unordered)
    mu = sort(mu);               % sort ascending

    mu_m = mu(end);              % largest eigenvalue
    mu_2 = mu(2);                % algebraic connectivity
end
