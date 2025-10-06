function [K, ris] = compute_KXMX(A, B, mu_2, mu_m)
%COMPUTE_KXMX Consensus coupling gain from LMI (inspired to Xu–Mo–Xie 2020, TAC).
%   [K, RIS] = COMPUTE_KXMX(A, B, MU_2, MU_M) computes the consensus gain K
%   by solving an LMI feasibility problem using YALMIP. The LMI structure is
%   adapted from:
%       L. Xu, Y. Mo, and L. Xie, "Distributed consensus over Markovian packet
%       loss channels," IEEE Trans. Autom. Control, vol. 65, no. 1, pp. 279–286,
%       Jan. 2020.
%
%   Inputs:
%     A     : (nx-by-nx) system/observer dynamic matrix used in the DKF internal model.
%     B     : (nx-by-nu) input/coupling matrix for consensus channels.
%     MU_2  : algebraic connectivity (second-smallest eigenvalue) of Laplacian Lg.
%     MU_M  : largest eigenvalue of Laplacian Lg.
%
%   Outputs:
%     K     : (nu-by-nx) consensus coupling gain, computed from the LMI solution.
%     RIS   : YALMIP optimizer result structure (from OPTIMIZE), useful for diagnostics.
%
%   Requirements:
%     - YALMIP (sdpvar/optimize/sdpsettings).
%     - An SDP solver supported by YALMIP (e.g., SeDuMi, MOSEK, SDPT3).
%
%   Notes:
%     • This routine sets p = 0 and q = 1 (as in a particular specialization
%       of the LMI), and defines c = 1 - ((mu_m - mu_2)/(mu_m + mu_2))^2 ∈ (0,1].
%     • Q1 and Q2 are scaled by trace constraints (trace(Q1)=trace(Q2)=1) to
%       avoid trivial solutions.
%     • K is formed post-hoc from the LMI solution as
%           K = -2/(mu_2 + mu_m) * inv(B' * inv(Q2) * B) * (B' * inv(Q2) *
%           A).
%
%   See also: SDPVAR, OPTIMIZE, SDPSETTINGS.

    % Dimensions
    [nx, nu] = size(B);

    % -------------------------- Decision variables --------------------------
    % Symmetric decision matrices for Lyapunov-like multipliers
    Q1 = sdpvar(nx);          % nx-by-nx, symmetric (YALMIP default with sdpvar(nx))
    Q2 = sdpvar(nx);          % nx-by-nx, symmetric
    % Full (non-symmetric) design matrices
    Z1 = sdpvar(nu, nx, 'full');
    Z2 = sdpvar(nu, nx, 'full');

    % --------------------------- Scalar parameters --------------------------
    % Following the chosen specialization of the LMI (see Xu–Mo–Xie 2020)
    p = 0;
    q = 1;

    % Connectivity-dependent factor c \in (0,1]
    % c = 1 - ((mu_m - mu_2)/(mu_m + mu_2))^2
    c = 1 - ((mu_m - mu_2)/(mu_m + mu_2))^2;

    % -------------------------- Block LMI constraints -----------------------
    % M1 >= 0
    M1 = [ Q1,                              (sqrt(q*c)*(A*Q1 + B*Z1))', (sqrt(q*(1-c))*A*Q1)',      (sqrt(1 - q)*A*Q1)';
           sqrt(q*c)*(A*Q1 + B*Z1),         Q2,                        zeros(nx, nx),              zeros(nx, nx);
           sqrt(q*(1-c))*A*Q1,              zeros(nx, nx),             Q2,                         zeros(nx, nx);
           sqrt(1 - q)*A*Q1,                zeros(nx, nx),             zeros(nx, nx),              Q1 ];

    % M2 >= 0
    M2 = [ Q2,                              (sqrt((1-p)*c)*(A*Q2 + B*Z2))', (sqrt((1-p)*(1-c))*A*Q2)', (sqrt(p)*A*Q2)';
           sqrt((1-p)*c)*(A*Q2 + B*Z2),     Q2,                            zeros(nx, nx),             zeros(nx, nx);
           sqrt((1-p)*(1-c))*A*Q2,          zeros(nx, nx),                 Q2,                        zeros(nx, nx);
           sqrt(p)*A*Q2,                    zeros(nx, nx),                 zeros(nx, nx),             Q1 ];

    % Collect constraints:
    %  - Semidefiniteness of the block LMIs
    %  - Trace normalization to fix scaling
    cnstr = [ M1 >= 0, M2 >= 0, trace(Q1) == 1, trace(Q2) == 1 ];

    % ------------------------------ Solve LMI -------------------------------
    options = sdpsettings('verbose', false);
    ris = optimize(cnstr, [], options);

    if (ris.problem ~= 0)
        % Nonzero problem code => solver/feasibility issue; keep going but warn.
        fprintf("Gamma computation WARN: Problem %d \n %s\n", ris.problem, ris.info);
    end

    % ------------------------- Gain recovery (post) -------------------------
    % Recover numeric Q2 and form K according to the chosen formula.
    % NOTE: for better conditioning you might prefer:
    %   K = -(2/(mu_2 + mu_m)) * ((B'*(Q2\B)) \ (B'*(Q2\A)));
    % but we keep the original explicit inv(...) structure here.
    Q2n = double(Q2);
    K = -2/(mu_2 + mu_m) * inv(B' * inv(Q2n) * B) * (B' * inv(Q2n) * A);
end
