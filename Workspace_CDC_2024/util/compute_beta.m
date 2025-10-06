function beta = compute_beta(Delta, A_u, n)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

options = optimoptions('fmincon','Display', 'iter',...
    'ConstraintTolerance', 1e-16, 'StepTolerance', 1e-12);
beta = fmincon(@(beta)(1), rand(n, 1), [], [], [], [], [], [],...
    @(beta)beta_constraints(beta, Delta, A_u, n), options);


end