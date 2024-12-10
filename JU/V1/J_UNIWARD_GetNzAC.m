function nzAC=J_UNIWARD_GetNzAC(coverPath)
%% Plots

C_STRUCT = jpeg_read(coverPath);

nzAC = nnz(C_STRUCT.coef_arrays{1})-nnz(C_STRUCT.coef_arrays{1}(1:8:end,1:8:end));