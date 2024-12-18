function [rhoB, rhoM1, rhoP1, nzAC] = J_UNIWARD_Loss_Center(coverPath)
%J_UNIWARD_LOSS_CENTER 此处显示有关此函数的摘要
%   此处显示详细说明
C_SPATIAL = double(imread(coverPath));
C_STRUCT = jpeg_read(coverPath);
C_COEFFS = C_STRUCT.coef_arrays{1};
C_QUANT = C_STRUCT.quant_tables{1};

wetConst = 10^13;
sgm = 2^(-6);

tStart = tic;% YY:analyze efficientcy

%% Get 2D wavelet filters - Daubechies 8
% 1D high pass decomposition filter
hpdf = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, ...
        -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768];
% 1D low pass decomposition filter
lpdf = (-1).^(0:numel(hpdf)-1).*fliplr(hpdf);

F{1} = lpdf'*hpdf;
F{2} = hpdf'*lpdf;
F{3} = hpdf'*hpdf;

% YY:
tEnd = toc(tStart);tStart = tic;
fprintf(1, '%.2fs[Get 2D wavelet filters - Daubechies 8]\n', tEnd);

%% Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
spatialImpact = cell(8, 8);
for bcoord_i=1:8
    for bcoord_j=1:8
        testCoeffs = zeros(8, 8);
        testCoeffs(bcoord_i, bcoord_j) = 1;
        spatialImpact{bcoord_i, bcoord_j} = idct2(testCoeffs)*C_QUANT(bcoord_i, bcoord_j);
    end
end

% YY:
tEnd = toc(tStart);tStart = tic;
fprintf(1, '%.2fs[Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1]\n', tEnd);


%% Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
waveletImpact = cell(numel(F), 8, 8);
for Findex = 1:numel(F)
    for bcoord_i=1:8
        for bcoord_j=1:8
            waveletImpact{Findex, bcoord_i, bcoord_j} = imfilter(spatialImpact{bcoord_i, bcoord_j}, F{Findex}, 'full');
        end
    end
end
% YY:
tEnd = toc(tStart);tStart = tic;
fprintf(1, '%.2fs[Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1]\n', tEnd);

%% Create reference cover wavelet coefficients (LH, HL, HH)
% Embedding should minimize their relative change. Computation uses mirror-padding
padSize = max([size(F{1})'; size(F{2})']);
C_SPATIAL_PADDED = padarray(C_SPATIAL, [padSize padSize], 'symmetric'); % pad image

RC = cell(size(F));
for i=1:numel(F)
    RC{i} = imfilter(C_SPATIAL_PADDED, F{i});
end

[k, l] = size(C_COEFFS);

nzAC = nnz(C_COEFFS)-nnz(C_COEFFS(1:8:end,1:8:end));
rho = 10000.*rand(k, l);
tempXi = cell(3, 1);
% YY:
tEnd = toc(tStart);tStart = tic;
fprintf(1, '%.2fs[Create reference cover wavelet coefficients (LH, HL, HH)]\n', tEnd);

%% Computation of costs
for row = round(k/4):round(3*k/4)
    for col = round(l/4):round(3*l/4)
        modRow = mod(row-1, 8)+1;
        modCol = mod(col-1, 8)+1;        
        
        subRows = row-modRow-6+padSize:row-modRow+16+padSize;
        subCols = col-modCol-6+padSize:col-modCol+16+padSize;
     
        for fIndex = 1:3
            % compute residual
            RC_sub = RC{fIndex}(subRows, subCols);            
            % get differences between cover and stego
            wavCoverStegoDiff = waveletImpact{fIndex, modRow, modCol};
            % compute suitability
            tempXi{fIndex} = abs(wavCoverStegoDiff) ./ (abs(RC_sub)+sgm);           
        end
        rhoTemp = tempXi{1} + tempXi{2} + tempXi{3};
        rho(row, col) = sum(rhoTemp(:));
    end
end

rhoM1 = rho;
rhoP1 = rho;
rhoB = rho;

rhoP1(rhoP1 > wetConst) = wetConst;
rhoP1(isnan(rhoP1)) = wetConst;    
rhoP1(C_COEFFS > 1023) = wetConst;
    
rhoM1(rhoM1 > wetConst) = wetConst;
rhoM1(isnan(rhoM1)) = wetConst;
rhoM1(C_COEFFS < -1023) = wetConst;

rhoB(rhoB > wetConst) = wetConst;
rhoB(isnan(rhoB)) = wetConst;    
rhoB(C_COEFFS > 1023) = wetConst;
rhoB(C_COEFFS < -1023) = wetConst;

% YY:
tEnd = toc(tStart);
fprintf(1, '%.2fs[Computation of costs]\n', tEnd);

