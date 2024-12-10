function [S_STRUCT, n_msg_bits, h] = J_UNIWARD_ED(coverPath, msg, rhoM1, rhoP1)

% -------------------------------------------------------------------------
% Copyright (c) 2013 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Contact: vojtech_holub@yahoo.com | fridrich@binghamton.edu | February
% 2013
%          http://dde.binghamton.edu/download/stego_algorithms/
% -------------------------------------------------------------------------

addpath('./STC');
C_STRUCT = jpeg_read(coverPath);
C_COEFFS = C_STRUCT.coef_arrays{1};


%% Embedding 
cover = int32(C_COEFFS(:)');
h = 10;      % constraint height - default is 10 - drives the complexity/quality tradeof
m = numel(msg); % number of message bits
nzAC = nnz(C_STRUCT.coef_arrays{1})-nnz(C_STRUCT.coef_arrays{1}(1:8:end,1:8:end));
alpha = m / (numel(C_COEFFS));
costs = zeros(3, numel(C_COEFFS), 'single'); % for each pixel, assign cost of being changed
costs(1,:) = rhoM1(:)';       % cost of changing by  -1
costs(3,:) = rhoP1(:)';       % cost of changing by  +1

tic;

[dist, stego, n_msg_bits,loss] = stc_pm1_pls_embed(cover, costs, msg, h);
fprintf('distortion per cover element = %f\n', dist / nzAC);
fprintf('        embedding efficiency = %f\n', alpha / (dist / nzAC));
fprintf('                  throughput = %1.1f Kbits/sec\n', numel(cover) / toc() / 1024);
fprintf('  achieved coding_loss = %4.2f%%\n', loss*100);

S_COEFFS = C_COEFFS;
S_COEFFS(:) = double(stego');
S_STRUCT = C_STRUCT;
S_STRUCT.coef_arrays{1} = S_COEFFS;

