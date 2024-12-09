close all;
clc;
I = imread('lena_512.bmp');
d = imread('logo_64.bmp');
data = reshape(d, 1, 64 * 64);

delta = 25.5;
stg = QIMHide(I, data);
transparency = psnr(I, stg);
figure;
subplot(1,2,1);imshow(I,[]);title('Org');
subplot(1,2,2);imshow(stg,[]);title(sprintf('PSNR:%.2f',transparency));

% extract without noise
msg = QIMDehide(I, stg, length(data));
m = reshape(msg, [64, 64]);
s = Similar(d,m);
figure;
subplot(1, 3, 1);imshow(d);title('watermark');
subplot(1, 3, 2);imshow(m);title(sprintf('%.2f',s));
subplot(1, 3, 3);imshow(xor(d, m));title('differ');

% extract with noise
dB = 35;
y = imnoise(stg, 'gaussian', 0, 10^(-dB/10));
transparency_noise = psnr(stg, y);
figure;
subplot(1,2,1);imshow(stg,[]);title(sprintf('PSNR:%.2f',transparency));
subplot(1,2,2);imshow(y,[]);title(sprintf('Gaussian:%.2f',transparency_noise));


msg = QIMDehide(I, y, length(data));
m = reshape(msg, [64, 64]);
sn = Similar(d,m);
figure;
subplot(1, 3, 1);imshow(d);title('watermark');
subplot(1, 3, 2);imshow(m);title(sprintf('%.2f',sn));
subplot(1, 3, 3);imshow(xor(d, m));title('differ');
