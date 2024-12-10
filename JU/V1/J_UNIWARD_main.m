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

clc;clear all; close all; fclose all;
addpath(fullfile('..','JPEG_Toolbox'));

%% Settings
payload = 0.2;   % measured in bits per non zero AC coefficients
filename = '01.jpg';
coverPath = fullfile('..','images_cover', filename);
stegoPath = fullfile('..','images_stego', filename);
nzAC=J_UNIWARD_GetNzAC(coverPath);
m = round(payload * nzAC); % number of message bits
d = imread("logo_64.bmp");
msg = reshape(d, 1, 64*64);
% msg = uint8(msg);
msg = uint8([msg zeros(1,m-length(msg))]);    


%% ------ compute loss with center
% Settings


% Alogrithm
fprintf('\nAlogrithm 4: center.--------------------\n');
tStart = tic;
nLevel = 1;
[~, rhoM1, rhoP1, ~] =  J_UNIWARD_Loss_Center(coverPath);
% fprintf("\n%d\n",rhoP1(1,1));
[S_STRUCT, n_msg_bits, h] = J_UNIWARD_ED(coverPath, msg, rhoM1, rhoP1);
jpeg_write(S_STRUCT, stegoPath);
% psnr
transparency = psnr(imread(coverPath), imread(stegoPath));
fprintf("\nPSNR: %.2f\n", transparency);
% noise
% dB = 35;
% y = imnoise(imread("../images_stego/01.jpg"), 'gaussian', 0, 10^(-dB/10));
% imwrite(y,stegoPath,"jpeg");
% 
extr_msg = J_UNIWARD_ET(stegoPath, n_msg_bits, h);
[~, wrong_rate] = biterr(extr_msg, msg);
fprintf("\nerror rate: %.2f\n", wrong_rate);
if all(extr_msg==msg)
    fprintf('Message was embedded and extracted correctly.\n');
    fprintf('  %d bits embedded => %d bits in 2LSB and %d bits in LSB.\n', ...
        sum(n_msg_bits), n_msg_bits(1), n_msg_bits(2));
end    
tEnd = toc(tStart);
fprintf('\nElapsed time: %.4f s, nLevel:%.2f\n', tEnd, nLevel);
fprintf('\nLoss: %.4f \n', log(J_UNIWARD_Distortion(coverPath,stegoPath)));
J_UNIWARD_Evaluate(coverPath,stegoPath);
% noise
dB = 35;
y = imnoise(imread("../images_stego/01.jpg"), 'gaussian', 0, 10^(-dB/10));
imwrite(y,stegoPath,"jpeg");
% 
extr_msg = J_UNIWARD_ET(stegoPath, n_msg_bits, h);
[~, wrong_rate] = biterr(extr_msg, msg);
fprintf("\nerror rate: %.2f\n", wrong_rate);

%% ------ compute loss with resample
fprintf('\nAlogrithm 3: resample.--------------------\n');
tStart = tic;
nLevel = 1;
[~, rhoM1, rhoP1, ~] =  J_UNIWARD_Loss_ResampleCost(coverPath, nLevel);
[S_STRUCT, n_msg_bits, h] = J_UNIWARD_ED(coverPath, msg, rhoM1, rhoP1);
jpeg_write(S_STRUCT, stegoPath);
% psnr
transparency = psnr(imread(coverPath), imread(stegoPath));
fprintf("\nPSNR: %.2f\n", transparency);
% noise
% dB = 35;
% y = imnoise(imread("../images_stego/01.jpg"), 'gaussian', 0, 10^(-dB/10));
% imwrite(y,stegoPath,"jpeg");
% 
extr_msg = J_UNIWARD_ET(stegoPath, n_msg_bits, h);
% [~, wrong_rate] = biterr(extr_msg, msg);
% fprintf("\nerror rate: %.2f\n", wrong_rate);
if all(extr_msg==msg)
    fprintf('Message was embedded and extracted correctly.\n');
    fprintf('  %d bits embedded => %d bits in 2LSB and %d bits in LSB.\n', ...
        sum(n_msg_bits), n_msg_bits(1), n_msg_bits(2));
end    
tEnd = toc(tStart);
fprintf('\nElapsed time: %.4f s, nLevel:%.2f\n', tEnd, nLevel);
fprintf('\nLoss: %.4f \n', log(J_UNIWARD_Distortion(coverPath,stegoPath)));
J_UNIWARD_Evaluate(coverPath,stegoPath);
    

 
% 
%% ------ compute loss with only one of the three filters
fprintf('\nAlogrithm 1.1: only LH is used.--------------------\n');
tStart = tic;
[~, rhoM1, rhoP1, ~] = J_UNIWARD_Loss_H(coverPath, 1); % only LH
[S_STRUCT, n_msg_bits, h] = J_UNIWARD_ED(coverPath, msg, rhoM1, rhoP1);
jpeg_write(S_STRUCT, stegoPath);
% psnr
transparency = psnr(imread(coverPath), imread(stegoPath));
fprintf("\nPSNR: %.2f\n", transparency);
% noise
% dB = 35;
% y = imnoise(imread("../images_stego/01.jpg"), 'gaussian', 0, 10^(-dB/10));
% imwrite(y,stegoPath,"jpeg");
% 
extr_msg = J_UNIWARD_ET(stegoPath, n_msg_bits, h);
% [~, wrong_rate] = biterr(extr_msg, msg);
% fprintf("\nerror rate: %.2f\n", wrong_rate);
if all(extr_msg==msg)
    fprintf('Message was embedded and extracted correctly.\n');
    fprintf('  %d bits embedded => %d bits in 2LSB and %d bits in LSB.\n', ...
        sum(n_msg_bits), n_msg_bits(1), n_msg_bits(2));
end
tEnd = toc(tStart);
fprintf('\nElapsed time: %.4f s\n', tEnd);
fprintf('\nLos: %.4f \n', log(J_UNIWARD_Distortion(coverPath,stegoPath)));
J_UNIWARD_Evaluate(coverPath,stegoPath);


%% ------ original version of loss computing
fprintf('\nAlogrithm 1.0: original version.--------------------\n');
tStart = tic;
[~, rhoM1, rhoP1, ~] = J_UNIWARD_Loss(coverPath);
[S_STRUCT, n_msg_bits, h] = J_UNIWARD_ED(coverPath, msg, rhoM1, rhoP1);
jpeg_write(S_STRUCT, stegoPath);
% psnr
transparency = psnr(imread(coverPath), imread(stegoPath));
fprintf("\nPSNR: %.2f\n", transparency);
% noise
% dB = 35;
% y = imnoise(imread("../images_stego/01.jpg"), 'gaussian', 0, 10^(-dB/10));
% imwrite(y,stegoPath,"jpeg");
% 
extr_msg = J_UNIWARD_ET(stegoPath, n_msg_bits, h);
% [~, wrong_rate] = biterr(extr_msg, msg);
% fprintf("\nerror rate: %.2f\n", wrong_rate);
if all(extr_msg==msg)
    fprintf('Message was embedded and extracted correctly.\n');
    fprintf('  %d bits embedded => %d bits in 2LSB and %d bits in LSB.\n', ...
        sum(n_msg_bits), n_msg_bits(1), n_msg_bits(2));
end
tEnd = toc(tStart);
fprintf('\nElapsed time: %.4f s\n', tEnd);
fprintf('\nLoss: %.4f \n', log(J_UNIWARD_Distortion(coverPath,stegoPath)));
J_UNIWARD_Evaluate(coverPath,stegoPath);


