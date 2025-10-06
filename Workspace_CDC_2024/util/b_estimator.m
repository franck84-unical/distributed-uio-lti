function b = b_estimator(y_i, y_i_hat)
%B_ESTIMATOR Estimate constant bias from measurement residuals.
%   B = B_ESTIMATOR(Y_I, Y_I_HAT) estimates a scalar bias term B that
%   minimizes the least-squares cost:
%
%       J(b) = sum_i ( y_i(i) - y_i_hat(i) - b )^2
%
%   Inputs:
%     Y_I     : vector of true/observed measurements.
%     Y_I_HAT : vector of predicted measurements (same length as Y_I).
%
%   Output:
%     B       : estimated constant bias (scalar).
%
%   Notes:
%     • Solved as a convex quadratic optimization using YALMIP.
%     • Solver set to 'gurobi'; change in SDPSETTINGS if unavailable.
%     • If RIS.PROBLEM ~= 0, a warning is printed and the result may be unreliable.

    % Decision variable (scalar bias)
    b = sdpvar(1);

    % Quadratic least-squares objective
    objfun = 0;
    for i = 1:length(y_i)
        objfun = objfun + (y_i(i) - y_i_hat(i) - b)' * (y_i(i) - y_i_hat(i) - b);
    end

    % No constraints in this problem
    cnstr = [];

    % Solve quadratic program
    options = sdpsettings('solver', 'gurobi', 'verbose', false, 'usex0', 0);
    ris = optimize(cnstr, objfun, options);

    % Warn if solver reports an issue
    if ris.problem ~= 0
        fprintf("Estimation WARN: Problem %d \n %s\n", ris.problem, ris.info);
    end

    % Return numeric bias
    b = double(b);
end
