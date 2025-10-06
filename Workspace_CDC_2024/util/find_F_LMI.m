function [Fi] = find_F_LMI(A, K, C, Lambda, i)
%FIND_F_LMI Solve for Fi in the DKF reconstruction equation via YALMIP.
%   FI = FIND_F_LMI(A, K, C, LAMBDA, I) returns the i-th block Fi (n-by-n)
%   such that:
%       Fi * Lambda = (A - K*C*A) * Fi
%       Fi * 1_n    = K(:, i)
%   where 1_n is the n-by-1 vector of ones.
%
%   Inputs:
%     A      : (n-by-n) state matrix.
%     K      : (n-by-m) steady-state Kalman gain.
%     C      : (m-by-n) output matrix used in the observer (not explicitly
%              used in constraints aside from forming A - K*C*A).
%     Lambda : (n-by-n) (typically diagonal) matrix of closed-loop modes.
%     i      : sensor/node index (1..m) selecting the i-th column of K.
%
%   Output:
%     Fi     : (n-by-n) solution of the linear matrix equalities.
%
%   Requirements:
%     - YALMIP (sdpvar/optimize/sdpsettings).
%     - A compatible LP/QP solver (here: 'gurobi' as specified).
%
%   Notes:
%     • The problem is posed as linear equalities in Fi (no objective).
%     • If the solver reports an issue (RIS.PROBLEM ~= 0), Fi is still
%       returned as DOUBLE(Fi) but a warning is printed.

    n = size(A, 1);
    Fi = sdpvar(n, n, 'full');

    % Linear equality constraints:
    %   Fi*Lambda = (A - K*C*A)*Fi
    %   Fi*ones(n,1) = K(:,i)
    M1 = [Fi*Lambda == (A - K*C*A)*Fi];
    M2 = [Fi*ones(n, 1) == K(:, i)];

    % Solve (no objective); keep original solver/verbosity choices
    options = sdpsettings('solver', 'gurobi', 'verbose', false);
    ris = optimize([M1, M2], [], options);

    Fi = double(Fi);

    if (ris.problem ~= 0)
        fprintf("F (i = %d) computation WARN: Problem %d \n %s\n", ...
                i, ris.problem, ris.info);
    end
end
