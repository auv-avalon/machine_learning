#!/usr/bin/env octave

NUMBERS = 1000

MU = [0 0]
SIGMA = [1 0.99; 0.99 1]

[mu_n, mu_m] = size(MU)
[S_n, S_m] = size(SIGMA)

[T p] = chol(SIGMA)

mu = MU(ones(NUMBERS, 1), :);

R = randn(NUMBERS, mu_m) * T + mu;

plot(R(:, 1), R(:, 2), "+1");
pause;


