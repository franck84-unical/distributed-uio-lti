classdef DistributedKalmanGainScheduling < handle
%DISTRIBUTEDKALMANGAINSCHEDULING Gain scheduling for a Distributed Kalman Filter (DKF).
%   This class precomputes and serves "vertices" (gain/structure sets) for a
%   distributed Kalman filtering scheme under sensor effectiveness changes
%   (quantized gamma parameters). Each vertex bundles:
%       - K:     steady-state Kalman gain for a given C(gamma)
%       - F:     node-specific reconstruction matrices
%       - S:     internal DKF dynamics matrix (Lambda + 1*beta')
%       - Lambda:closed-loop eigenvalues for A - K*C*A
%       - beta:  augmentation vector (couples unstable A_u, if provided)
%       - Gamma: consensus coupling matrix (from graph Laplacian)
%
%   Typical usage:
%       dkf = DistributedKalmanGainScheduling(A, A_u, C_t, Q, R, n_gamma, ...
%                                             eps_, Lg, steps);
%       [F, K, C, S, Lambda, beta, Gamma] = dkf.get_vertices(p_quantized);
%
%   Notes:
%     • The constructor precomputes vertices for all admissible
%       quantized gamma combinations (controlled by 'steps'). Only
%       observable configurations are stored.
%     • Switching interpolation code exists but is currently bypassed
%       by an early RETURN in GET_VERTICES (i.e., it outputs the target
%       vertex immediately). See comments in GET_VERTICES if you plan
%       to enable smooth transitions.
%
%   See also: IDARE, OBSV, EIGS, FIND_EIG (user util), FIND_F_LMI (user util),
%             COMPUTE_BETA (user util), COMPUTE_KXMX (user util)

    %============================== Properties ==============================%
    properties
        % vertices : containers.Map -> key: char(num2str(p)), value: struct with
        %             fields {K, C, F, Lambda, beta, S, Gamma} for a specific
        %             quantized gamma vector p (1-by-n_gamma).
        vertices

        % eps_ : consensus tuning parameter (scalar, >0), used for design checks.
        eps_

        % Lg : graph Laplacian (m-by-m) of the communication network.
        Lg

        % sigmoid : @(u) smooth step used for optional gain interpolation
        %           during switching (see GET_VERTICES). By default, a
        %           logistic function centered at 'end_time'.
        sigmoid

        % old_p : previous gamma vector (1-by-n_gamma), used for switching.
        old_p

        % old_vertex : cached vertex struct associated with old_p.
        old_vertex

        % next_p : target gamma vector during a switch (1-by-n_gamma).
        next_p

        % next_vertex : vertex struct associated with next_p.
        next_vertex

        % counter : integer "time" used by the sigmoid schedule.
        counter

        % switching_mode : boolean flag enabling smooth interpolation of
        %                  gains/states between vertices.
        switching_mode

        % end_time : center of the sigmoid time window (scalar > 0).
        end_time

        % steps : vector of admissible quantization levels for each gamma.
        steps
    end

    %=============================== Methods ================================%
    methods
        function obj = DistributedKalmanGainScheduling(A, A_u, C, Q, R, n, eps_, Lg, steps, c_perm)
        %DISTRIBUTEDKALMANGAINSCHEDULING Construct the scheduler and precompute vertices.
        %
        %   OBJ = DistributedKalmanGainScheduling(A, A_u, C, Q, R, n, eps_, Lg, steps)
        %   builds a collection of vertices for all admissible quantized gamma
        %   combinations based on the quantization grid 'steps'.
        %
        %   Inputs:
        %     A      : (nx-by-nx) state matrix.
        %     A_u    : (nx_u-by-nx_u) unstable sub-block of A (may be empty []).
        %     C      : function handle C(p) returning the output matrix for a given
        %              gamma vector p (1-by-n). Example: C = @(p) C_t(p).
        %     Q      : (nx-by-nx) process noise covariance (for design in IDARE).
        %     R      : (m-by-m) measurement noise covariance (for design in IDARE).
        %     n      : number of gamma parameters (length of p).
        %     eps_   : consensus scaling parameter (scalar > 0), used downstream
        %              in stability/connectivity checks and Gamma design.
        %     Lg     : (m-by-m) graph Laplacian of the communication graph.
        %     steps  : row vector of quantization levels for each gamma (e.g., [0 0.3 0.6 1]).
        %     c_perm : (optional) explicit set of gamma combinations to use
        %              instead of enumerating all combinations from 'steps'.
        %              If provided, each row is a 1-by-n gamma vector.
        %
        %   Behavior:
        %     • If C(p) leads to an observable pair (A, C(p)), a vertex is computed
        %       via COMPUTE_VERTEX and stored in OBJ.VERTICES keyed by num2str(p).
        %     • If c_perm is not provided, all combinations of 'steps' of length n
        %       (excluding the first index interpretation per find_all_perm) are tried.
        %
        %   See also: COMPUTE_VERTEX, OBSV, IDARE.

            obj.eps_ = eps_;
            obj.Lg   = Lg;

            % Sigmoid used for optional smooth switching
            bet = 5;              % slope
            obj.end_time = 7;     % center
            obj.sigmoid = @(u) 1 - 1./(1 + exp(-bet*(u - obj.end_time)));

            obj.vertices = containers.Map('KeyType', 'char', 'ValueType', 'any');
            nx = size(A, 1);
            m  = size(C(ones(n, 1)), 1);  % infer #outputs from C evaluated at p = ones(n,1)
            obj.steps = steps;

            % Enumerate gamma combinations
            if (nargin < 10)
                % find_all_perm(k, n) is assumed to be a user utility that returns
                % all index combinations for k=(length(steps)-1) levels across n gammas.
                all_perm = find_all_perm(length(steps) - 1, n);
            else
                all_perm = c_perm;
            end

            % Build vertices for observable configurations
            for i = 1:size(all_perm, 1)
                if (nargin < 10)
                    p = steps(1 + all_perm(i, :));  % map indices to actual step values
                else
                    p = all_perm(i, :);             % already given as values
                end
                C_v = C(p);
                % Store only if (A,C_v) observable and key not yet present
                if (rank(obsv(A, C_v)) == nx) && ~isKey(obj.vertices, num2str(p))
                    disp(p); %#ok<DISP>
                    vertex = obj.compute_vertex(A, C_v, Q, R, A_u, m, nx);
                    obj.vertices(num2str(p)) = vertex;
                end
            end

            % Initialize switching-related state (even if switching is bypassed)
            obj.switching_mode = false;
            obj.next_vertex    = obj.vertices(num2str(ones(1, n)));
            obj.old_vertex     = obj.vertices(num2str(ones(1, n)));
            obj.old_p          = ones(1, n);
            obj.next_p         = p;   %#ok<NASGU> % last enumerated p; kept for symmetry
            obj.counter        = 0;
        end

        function [c, ceq] = find_Jorndan_c(obj, p, A, K, C)
        %FIND_JORDNAN_C (likely "find_Jordan_c") helper for pole-matching constraints.
        %   [C, CEQ] = FIND_JORDNAN_C(OBJ, P, A, K, Cmat) builds inequality and
        %   equality constraints intended to match the closed-loop characteristic
        %   polynomial to that of a (block) Jordan structure Lambda built from P.
        %
        %   Inputs:
        %     p     : vector used on the diagonal of Lambda.
        %     A     : state matrix.
        %     K     : Kalman gain matrix.
        %     Cmat  : output matrix (same size as in observer design).
        %
        %   Outputs:
        %     c   : inequality constraint vector (here eigs(Lambda) - 1 < 0).
        %     ceq : equality constraints vector enforcing charpoly(A - K*C*A)
        %           == charpoly(Lambda).
        %
        %   Note:
        %     • The method name has a typographical error retained for
        %       backward compatibility. No functional change is applied.

            %#ok<*INUSD> inputs unused except where referenced by design
            Lambda = diag(p);
            for i = 1:size(Lambda,1) - 1
                Lambda(i, i+1) = 1;    % build (upper) Jordan-like structure
            end
            c   = eigs(Lambda) - 1;    % enforce spectral radius < 1
            ceq = reshape(charpoly(A - K*C*A) - charpoly(Lambda), [], 1);
        end

        function vertex = compute_vertex(obj, A, C_v, Q, R, A_u, m, nx)
        %COMPUTE_VERTEX Build a DKF vertex for a specific output matrix C_v.
        %   VERTEX = COMPUTE_VERTEX(OBJ, A, C_v, Q, R, A_u, m, nx) computes:
        %     - K      : steady-state Kalman gain via IDARE
        %     - Lambda : diag of eigenvalues of (A - K*C_v*A)
        %     - F      : block matrix [F1 ... Fm] solving Fi*Lambda = (A - K*C_v*A)*Fi
        %     - beta   : augmentation vector (zeros if A_u is empty)
        %     - S      : Lambda + 1*beta'
        %     - Gamma  : consensus coupling matrix from S and graph spectrum
        %
        %   Assumptions:
        %     • (A, C_v) is observable (checked before calling).
        %     • User utilities FIND_F_LMI, COMPUTE_BETA, COMPUTE_KXMX, FIND_EIG exist.
        %     • EIGS is used for speed; the system is small/moderate.
        %
        %   See also: IDARE, EIGS.

            % Steady-state Kalman gain (dual ARE formulation)
            [~, K] = idare(A', C_v', Q, R, [], []);
            K = K';

            % Closed-loop eigenvalues of A - K*C_v*A
            % (diagonal of eigenvalues, real part only)
            Lambda = real(diag(eigs((A - K*C_v*A))));
            assert(all(eigs(Lambda) < 1), 'Unstable Lambda detected.');

            % Build F = [F1 ... Fm], each Fi solves Fi*Lambda = (A - K*C_v*A)*Fi
            F = zeros(nx, nx*m);
            for i = 1:m
                Fi = find_F_LMI(A, K, C_v, Lambda, i);
                assert(all(abs(Fi*Lambda - (A - K*C_v*A)*Fi) < 1e-6, 'all'), ...
                       'Fi equation residual too large.');
                F(:, ((i-1)*nx + 1):(i)*nx) = Fi;
            end

            % beta augmentation (if A_u provided), otherwise zeros
            if ~isempty(A_u)
                beta_v = compute_beta(Lambda, A_u, nx);
                % Examples of alternatives the author experimented with:
                % beta_v = place(A, ones(nx, 1), [1, 0.5, 0.6])';  % comment in original
            else
                beta_v = zeros(nx, 1);
            end

            % Internal DKF dynamics + consensus coupling
            S = Lambda + ones(nx, 1)*beta_v';
            [mu_m, mu_2] = find_eig(obj.Lg);            % graph spectral bounds
            Gamma = -compute_KXMX(S, ones(nx, 1), mu_2, mu_m);

            vertex = struct('K', K, 'C', C_v, 'F', F, 'Lambda', Lambda, ...
                            'beta', beta_v, 'S', S, 'Gamma', Gamma);
        end

        function [F, K, C_out, S, Lambda, beta, Gamma] = get_vertices(obj, p)
        %GET_VERTICES Retrieve the vertex for a given quantized gamma vector.
        %   [F,K,C,S,Lambda,beta,Gamma] = GET_VERTICES(OBJ, P) returns the
        %   precomputed vertex associated with the quantized gamma vector P.
        %
        %   Input:
        %     p : (n-by-1 or 1-by-n) quantized gamma vector (entries must match
        %         one of the keys used during precomputation).
        %
        %   Outputs:
        %     F, K, C, S, Lambda, beta, Gamma : fields of the stored vertex.
        %
        %   Notes on switching:
        %     • The code below currently bypasses smooth interpolation by returning
        %       NEXT_VERTEX immediately (and 'beta','Gamma' from OLD_VERTEX as in the
        %       original implementation). If you want smooth transitions, remove the
        %       early RETURN and use the interpolation section that follows.

            if(~isrow(p))
                p = p';
            end

            % Immediate (non-smoothed) switching policy:
            obj.next_vertex = obj.vertices(num2str(p));
            F      = obj.next_vertex.F;
            K      = obj.next_vertex.K;
            C_out  = obj.next_vertex.C;
            S      = obj.next_vertex.S;
            Lambda = obj.next_vertex.Lambda;

            % Keep beta and Gamma from the previous vertex (as per original code)
            beta   = obj.old_vertex.beta;
            Gamma  = obj.old_vertex.Gamma;
            return

            % ------------------ Smooth switching (currently bypassed) ------------------
            % if ~isequal(obj.old_p, p) && ~obj.switching_mode
            %     obj.switching_mode = true;
            %     obj.next_vertex = obj.vertices(num2str(p));
            %     obj.old_vertex  = obj.vertices(num2str(obj.old_p));
            %     obj.next_p      = p;
            %     obj.counter     = 0;
            %     disp("switching");
            % end
            %
            % if obj.switching_mode
            %     s    = obj.sigmoid(obj.counter);
            %     F      = (obj.old_vertex.F      - obj.next_vertex.F     )*s + obj.next_vertex.F;
            %     K      = (obj.old_vertex.K      - obj.next_vertex.K     )*s + obj.next_vertex.K;
            %     C_out  = (obj.old_vertex.C      - obj.next_vertex.C     )*s + obj.next_vertex.C;
            %     S      = (obj.old_vertex.S      - obj.next_vertex.S     )*s + obj.next_vertex.S;
            %     Lambda = (obj.old_vertex.Lambda - obj.next_vertex.Lambda)*s + obj.next_vertex.Lambda;
            %     beta   = (obj.old_vertex.beta   - obj.next_vertex.beta  )*s + obj.next_vertex.beta;
            %     Gamma  = obj.old_vertex.Gamma;  % or interpolate if desired
            %     obj.counter = obj.counter + 1;
            % else
            %     F      = obj.old_vertex.F;
            %     K      = obj.old_vertex.K;
            %     C_out  = obj.old_vertex.C;
            %     S      = obj.old_vertex.S;
            %     Lambda = obj.old_vertex.Lambda;
            %     beta   = obj.old_vertex.beta;
        end
    end
end