function [all_perm] = find_all_perm(n, m)
%FIND_ALL_PERM Generate all integer combinations from 0..n of length m.
%   ALL_PERM = FIND_ALL_PERM(N, M) returns a matrix of size
%   ((N+1)^M)-by-M, where each row is a unique combination of
%   integers in {0,1,...,N} of length M.
%
%   Example:
%       >> find_all_perm(1, 2)
%       ans =
%            0     0
%            1     0
%            0     1
%            1     1
%
%   Used for enumerating quantized gamma vectors in DKF scheduling.
%
%   Inputs:
%     N : maximum integer (0..N).
%     M : number of positions.
%
%   Output:
%     ALL_PERM : ((N+1)^M)-by-M array of all integer combinations.

    % Preallocate result
    all_perm = zeros((n+1)^m, m);

    % Simple "odometer" style enumeration
    for i = 2:(n+1)^m
        j = 1;
        all_perm(i, :) = all_perm(i - 1, :);
        while (j <= m)
            if all_perm(i, j) == n
                all_perm(i, j) = 0;
            else
                all_perm(i, j) = all_perm(i, j) + 1;
                break;
            end
            j = j + 1;
        end
    end
end
