function J_UNIWARD_Evaluate(coverPath,stegoPath)
%% Plots

S_STRUCT = jpeg_read(stegoPath);
C_STRUCT = jpeg_read(coverPath);
C_SPATIAL = double(imread(coverPath));
S_SPATIAL = double(imread(stegoPath));

nzAC = nnz(C_STRUCT.coef_arrays{1})-nnz(C_STRUCT.coef_arrays{1}(1:8:end,1:8:end));
fprintf('\nchange rate per nzAC: %.4f, nzAC: %d\n', sum(S_STRUCT.coef_arrays{1}(:)~=C_STRUCT.coef_arrays{1}(:))/nzAC, nzAC);

%% Plots
figure;
imshow(uint8(C_SPATIAL));
title('Cover image');

figure;
imshow(uint8(S_SPATIAL));
title('Stego image');

figure; 
diff = S_STRUCT.coef_arrays{1}~=C_STRUCT.coef_arrays{1};
imshow(diff);
title('Changes in DCT domain (in standard JPEG grid)');

figure;
diff = S_SPATIAL-C_SPATIAL;
cmap = colormap('Bone');
imshow(diff, 'Colormap', cmap);
c=colorbar('Location', 'SouthOutside');
caxis([-20, 20]);
title('Changes in spatial domain caused by DCT embedding');