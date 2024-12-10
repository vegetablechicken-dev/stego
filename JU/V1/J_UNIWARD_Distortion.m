function y = J_UNIWARD_Distortion(coverPath,stegoPath)

sgm = 2^(-6);

C_SPATIAL = double(imread(coverPath));
S_SPATIAL = double(imread(stegoPath));


%% Get 2D wavelet filters - Daubechies 8
% 1D high pass decomposition filter
hpdf = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, ...
        -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768];
% 1D low pass decomposition filter
lpdf = (-1).^(0:numel(hpdf)-1).*fliplr(hpdf);

F{1} = lpdf'*hpdf;
F{2} = hpdf'*lpdf;
F{3} = hpdf'*hpdf;

%% Create reference cover wavelet coefficients (LH, HL, HH)
% Embedding should minimize their relative change. Computation uses mirror-padding
padSize = max([size(F{1})'; size(F{2})']);
C_SPATIAL_PADDED = padarray(C_SPATIAL, [padSize padSize], 'symmetric'); % pad image
S_SPATIAL_PADDED = padarray(S_SPATIAL, [padSize padSize], 'symmetric');

RC = cell(size(F));
RS = cell(size(F));
temp_loss = cell(size(F));
loss = zeros(1, 3);
for i=1:numel(F)
    RC{i} = imfilter(C_SPATIAL_PADDED, F{i});
    RS{i} = imfilter(S_SPATIAL_PADDED, F{i});
%     loss(i) = abs(sum(RC{i}(:) - RS{i}(:)))/sum(abs(RC{i}(:)));
    temp_loss{i} = abs(RC{i} - RS{i}) ./ (abs(RC{i})+sgm);
end
temp = temp_loss{1} + temp_loss{2} + temp_loss{3};
% y = sum(loss);
y = sum(temp(:));