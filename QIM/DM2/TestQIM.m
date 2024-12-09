I = imread('lena_512.bmp');
d = imread('logo_64.bmp');
data = reshape(d, 1, 64 * 64);

delta_values = 10:0.5:100;
psnr_values = zeros(size(delta_values));
s_no_noise = zeros(size(delta_values));
s_noise = zeros(size(delta_values));

for i = 1 : length(delta_values)
    delta = delta_values(i);
    stg = QIMHide(I, data, delta);
    % PSNR
    transparency = psnr(I, stg);
    psnr_values(i) = transparency;
    % extract without noise
    msg = QIMDehide(stg, delta, length(data));
    m = reshape(msg, [64, 64]);
    s = Similar(d,m);
    s_no_noise(i) = s;
    % extract with noise
    dB = 35;
    y = imnoise(stg, 'gaussian', 0, 10^(-dB/10));
    msg = QIMDehide(y, delta, length(data));
    m = reshape(msg, [64, 64]);
    sn = Similar(d,m);
    s_noise(i) = sn;
end

figure;
plot(delta_values, psnr_values);
xlabel('Delta Value');
ylabel('PSNR (dB)');
title('PSNR vs. Delta for QIM');

figure;
plot(delta_values, s_no_noise);
xlabel('Delta Value');
ylabel('Similar without Noise');
title('Similar vs. Delta for QIM without Noise');

figure;
plot(delta_values, s_noise);
xlabel('Delta Value');
ylabel('Similar with Noise');
title('Similar vs. Delta for QIM with Noise');

