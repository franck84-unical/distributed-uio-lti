% =========================================================================
%  © 2023 University of Calabria — All rights reserved
%
%  Title: Distributed Kalman Filter with Gain Scheduling on a Ring Network
%
%  Description:
%    Demonstration script for simulating a Distributed Kalman Filter (DKF)
%    with gain scheduling in the presence of faulty sensors
%    (bias and varying effectiveness), and communication over a cycle graph
%    with consensus.
%
%    Results published in F. A. Torchiaro, G. Gagliardi, F. Tedesco, 
%    A. Casavola and B. Sinopoli, A Fault-Tolerant Distributed Sensor 
%    Reconciliation Scheme based on Decomposed Steady-State Kalman Filter,
%    2024 IEEE 63rd Conference on Decision and Control (CDC), Milan, Italy,
%    2024.
%
%  Usage notes:
%    - Requires the folder ./util in the MATLAB path (helper functions such
%      as find_eig, b_estimator, and the dkf object/class with get_vertices).
%    - If available, precomputed DKF gains are loaded from "prestored_dkf.mat".
%    - Variables ending with *_t denote logged trajectories.
%    - Intended for academic and didactic use only.
%    - The toolbox Yalmip and the solver Gurobi are required.
% =========================================================================

clear; close all; clc;

%% Include util folder
addpath(genpath('./util'));

%% Plant and Parameters Setup
nx = 2;                         % Number of states
nu = 2;                         % Number of inputs
m  = 5;                         % Number of sensors/nodes
n_gamma = 3;                    % Number of effectiveness parameters (gamma)

steps = [0, 0.3, 0.6, 1];       % Quantization levels for gamma (rho = 0.3)

tc = 0.1;                       % Sampling time [s]

A   = [0.4, 0.6; 0.3, 0.3];     % State transition matrix
A_u = [];                       % Unstable part (if not used, leave empty)
B   = eye(nx, nu);              % Input matrix

% Output matrix with faults (parametric in gamma)
%   gamma(1), gamma(2), gamma(3) ∈ [0,1] model sensor effectiveness
C_t = @(gamma) ([gamma(1)     0;
                  0      gamma(2);
                  gamma(3)  gamma(3);
                  0         2; ...
                 -1         0 ]);

x0 = 2*rand(nx, 1);             % Random initial state
Im = eye(m); 
Inx = rand(nx); 

%% Steady-State Kalman Filter Parameters (Centralized)
Q = 0.25*eye(nx);
R = 0.001*eye(m);               % Measurement noise covariance

%% Communication Graph (Cycle: 1–2–3–4–5–1)
g = graph();
g = g.addnode(m);
g = g.addedge(1, 2);
g = g.addedge(2, 3);
g = g.addedge(3, 4);
g = g.addedge(4, 5);
g = g.addedge(5, 1);

figure(2); clf;
plot(g, "MarkerSize", 7);
title('Communication graph (cycle)');
axis equal;

adj_matrix = full(g.adjacency());       % Adjacency matrix
Lg = laplacian(g);                      % Graph Laplacian
[mu_m, mu_2] = find_eig(Lg);            % Smallest nonzero and largest eigenvalues

% Consensus parameter
eps_ = 0.5;

% Connectivity/robustness condition (only if A_u is defined)
if ~isempty(A_u)
    assert(...
        prod(eigs(A_u, size(A_u, 1))) < 1/eps_ && 1/eps_ <= (1 + mu_2/mu_m)/(1 - mu_2/mu_m), ...
        'Connectivity/robustness condition not satisfied.'...
    );
else
    % If A_u is empty, skip assertion.
end

%% Load Precomputed DKF Gains
%load("prestored_dkf.mat");      % Loads dkf object/structure with get_vertices(...)
% Alternative (potentially expensive):
dkf = DistributedKalmanGainScheduling(A, A_u, C_t, Q, R, n_gamma, eps_, Lg, steps);

%% Simulation Parameters and Preallocation
x_hat     = cell(m, 1);         % Local state estimates for each node
p_hat_t   = cell(m, 1);         % Quantized gamma estimates (log)
p_hat_c_t = cell(m, 1);         % Continuous gamma estimates (log)
for i = 1:m
    x_hat{i}     = zeros(nx, 1);
    p_hat_t{i}   = ones(n_gamma, 2);
    p_hat_c_t{i} = ones(n_gamma, 2);
end

p_t         = zeros(n_gamma, 2);    % True gamma trajectory
forced_resp = zeros(nx, m);         % Forced response at each node
u_t         = [0, 0]';              % Input log
p           = [1, 1, 1]';           % True gamma (sensor effectiveness)
b_hat       = zeros(m, 1);          % Bias estimate (nodes 1 and 2)
N           = 5;                    % Bias estimation window length
b           = zeros(m, 1);          % True bias
b_t         = b;                    % Bias log
y_t         = C_t(p)*x0;            % Measurement log
y_hat_t     = cell(m, 1);
for i = 1:m
    y_hat_t{i} = y_t(i);
end

% DKF / consensus variables
zetak     = zeros(nx, m);
etak      = zeros(m*(nx), m);
Delta_k   = cell(m, m);
Delta_kp1 = cell(m, m);
for i = 1:m
    for j = 1:m
        Delta_k{i, j}   = zeros(m*(1), 1);
        Delta_kp1{i, j} = zeros(m*(1), 1);
    end
end

z         = zeros(m, 1);            % Local innovations
yk        = zeros(m, 1);            % Current measurements
b_hat_t   = zeros(m, 1);            % Bias estimate log

t  = 0;
x  = x0;
xk = x0;

%% Simulation Loop
% 180 steps (18 s with tc = 0.1 s)
for k = 1:180

    %%%%%%%%%%%%%%%%%%%%% Fault Injection %%%%%%%%%%%%%%%%%%%%%
    % Bias (appears after enough samples are collected)
    if size(x_hat{i}, 2) > 60
        b(1) = sin(2*pi/160*(k - 60));
        b(2) = sin(2*pi/160*(k - 60));
    end

    % Sensor effectiveness (gamma)
    if k > 40
        p(1) = 1 - exp(-24/(k-39));
        p(2) = 1 - exp(-24/(k-39));
    end
    if k > 80
        p(3) = 1 - exp(-12/(k-79));
    end
    if k > 96
        p(3) = 0.5;
    end
    if k > 145
        p(1) = 0;
        p(2) = 0;
    end

    p_t = [p_t, p];         % Log true gamma
    b_t = [b_t, b];         % Log true bias
    C   = C_t(p);           % Current output matrix
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%% Measurement Noise %%%%%%%%%%%%%%%%%%%
    v(:, k) = mvnrnd(zeros(m, 1), R); %#ok<*SAGROW>  % v(k) ~ N(0, R)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % System dynamics (sinusoidal input)
    u  = [sin(t(end)); cos(t(end))];
    xk = A*xk + B*u;                % Next state (no process noise here)
    yk = C*xk + b + v(:, k);        % Measurement with bias + noise

    % Logging
    y_t = [y_t yk];
    t   = [t; t(end) + tc];
    x   = [x, xk];

    % Filter update (for each node)
    for i = 1:m
        % Scheduled configuration (quantized gamma estimate)
        p_hat_t_i        = p_hat_t{i};
        forced_resp(:, i)= A*forced_resp(:, i) + B*u;
        C_hat            = C_t(p_hat_t_i(:, end));

        % Vertices/parameters from DKF
        [F, K, C_test, S, Lambda, beta, Gamma] = dkf.get_vertices(p_hat_t_i(:, end)); %#ok<ASGLU>
        Fi = F(:, ((i-1)*nx + 1):(i)*nx); %#ok<NASGU>

        % Local innovation
        yk_n     = yk(i) - C_hat(i, :)*forced_resp(:, i) - b_hat(i);
        z(i)     = yk_n - beta'*zetak(:, i);
        zetak(:, i) = S*zetak(:, i) + ones(nx, 1)*z(i);

        % --- Consensus update (1 step per iteration) ---
        S_tilde     = kron(eye(m), S);
        L_tilde     = kron(eye(m), ones(nx, 1));
        B_tilde     = kron(eye(m), ones(nx, 1));
        Gamma_tilde = kron(eye(m), Gamma);
        steps_cons  = 1;
        for step = 1:steps_cons %#ok<NASGU>
            u_c = zeros(m*(1), 1);
            Delta_k{i, i} = Gamma_tilde*etak(:, i);
            for j = 1:m
                u_c = u_c + adj_matrix(i, j)*(Delta_k{i, j} - Delta_k{i, i});
            end
            etak(:, i) = S_tilde*etak(:, i) + (1/steps_cons)*L_tilde(:, i)*z(i) + B_tilde*u_c;
            for j = 1:m
                if adj_matrix(j, i) > 0
                    Delta_kp1{j, i} = Gamma_tilde*etak(:, i);
                end
            end
        end
        % ---------------------------------------------------

        % State estimate retrieval
        x_hat_i = m*F*etak(:, i) + forced_resp(:, i);

        % Log estimates
        x_hat{i} = [x_hat{i}, x_hat_i];

        % Update gamma estimates (continuous + quantized)
        p_hat_c_i = p;                                 % continuous proxy
        [~, idx]  = min(abs(p_hat_c_i' - steps.'));    % quantization
        p_hat_i   = steps(idx)';                       % quantized estimate
        y_hat_t{i}= [y_hat_t{i}, C(i, :)*x_hat_i];

        % Bias estimation (for nodes 1 and 2, after a transient)
        if size(x_hat{i}, 2) > 30 && (i == 1 || i == 2)
            b_hat(i) = b_estimator(y_t(i, end-N:end),  y_hat_t{i}(end-N:end));
        end

        p_hat_t{i}   = [p_hat_t{i},   p_hat_i];
        p_hat_c_t{i} = [p_hat_c_t{i}, p_hat_c_i];
    end

    % Consensus propagation
    Delta_k = Delta_kp1;

    % Log input and bias estimate
    u_t     = [u_t, u];
    b_hat_t = [b_hat_t, b_hat];
end

%% Plots: states, inputs, and gamma
figure(1); clf;
sgtitle('State estimation, gamma parameters, and inputs');

for j = 1:nx+2
    subplot(nx + 2, 1, j);
    grid on; hold on;

    if j < nx + 1
        % States
        title(sprintf('x_%d', j));
        xlabel('t [s]'); ylabel(sprintf('x_%d', j));
        plot(t, x(j, :), 'LineWidth', 2, 'DisplayName', 'true');
        for node = 1:m
            x_hat_i = x_hat{node};
            plot(t, x_hat_i(j, :), '--', 'LineWidth', 1.5, ...
                 'DisplayName', sprintf('\\hat{x}_{%d}^{(node %d)}', j, node));
        end
        if j == 1
            %legend('Location','bestoutside');
        end

    elseif j == nx + 2
        % Inputs
        title('Input u');
        xlabel('t [s]'); ylabel('u');
        plot(t, u_t(1, :), 'LineWidth', 2, 'DisplayName','u_1');
        plot(t, u_t(2, :), 'LineWidth', 2, 'DisplayName','u_2');
        %legend('Location','best');

    else
        % Gamma parameters
        title('\gamma (true vs estimated)');
        xlabel('t [s]'); ylabel('\gamma');
        for node = 1:m
            p_hat_c_t_i = p_hat_t{node};
            for jj = 1:length(p)
                plot(t, p_t(jj, 2:end),            'LineWidth', 2,  'DisplayName', sprintf('\\gamma_{%d} true', jj));
                plot(t, p_hat_c_t_i(jj, 2:end), '--', 'LineWidth', 1.5, 'DisplayName', sprintf('\\hat{\\gamma}_{%d} (node %d)', jj, node));
            end
        end
        %legend('Location','bestoutside');
    end
end

%% Plots: bias estimation (nodes 1 and 2)
figure(3); clf;
for j = [1, 2]
    subplot(2, 1, j);
    grid on; hold on;
    title(sprintf('$b_%d(k)$', j), 'Interpreter','latex');
    xlabel('t [s]', 'Interpreter','latex');
    ylabel(sprintf('$b_%d(k)$', j), 'Interpreter','latex');

    plot(t, b_t(j, :),     'LineWidth', 2,  'DisplayName', sprintf('$b_%d(k)$', j));
    plot(t, b_hat_t(j, :), '--',        'LineWidth', 2,  'DisplayName', sprintf('$\\hat b_%d(k)$', j));
    %legend('Interpreter','latex', 'Location','best');
end
drawnow;
