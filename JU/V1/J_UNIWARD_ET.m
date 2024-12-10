function extr_msg = J_UNIWARD_ET(stegoPath, n_msg_bits, h)

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
% This function simulates embedding using J-UNIWARD steganographic 
% algorithm.
addpath('./STC');
S_STRUCT = jpeg_read(stegoPath);
S_COEFFS = S_STRUCT.coef_arrays{1};


%% Embedding 
stego = int32(S_COEFFS(:)');
tic;
extr_msg = stc_ml_extract(stego, n_msg_bits, h);