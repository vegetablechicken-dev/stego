I = imread('lena_512.bmp');
d = imread('logo_64.bmp');
data = reshape(d, 1, 64 * 64);

delta = 10;
stg1 = QIMHide(I, data, delta);
transparency1 = psnr(I, stg1);

delta = 100;
stg2 = QIMHide(I, data, delta);
transparency2 = psnr(I, stg2);

figure;
subplot(1,3,1);imshow(I,[]);title('Org');
subplot(1,3,2);imshow(stg1,[]);title(sprintf('Delta:10; PSNR:%.2f',transparency1));
subplot(1,3,3);imshow(stg2,[]);title(sprintf('Delta:100; PSNR:%.2f',transparency2));

dB = 35;
y = imnoise(stg2, 'gaussian', 0, 10^(-dB/10));
msg = QIMDehide(y, delta, length(data));
m = reshape(msg, [64, 64]);
sn = Similar(d,m);
figure;
subplot(1,3,1);imshow(m);title(sprintf('delta:%d, similar: %.2f',delta, sn));
delta = 10;
y = imnoise(stg1, 'gaussian', 0, 10^(-dB/10));
msg = QIMDehide(y, delta, length(data));
m = reshape(msg, [64, 64]);
sn = Similar(d,m);
subplot(1,3,3);imshow(m);title(sprintf('delta:%d, similar: %.2f',delta, sn));

delta = 20;
stg = QIMHide(I, data, delta);
y = imnoise(stg, 'gaussian', 0, 10^(-dB/10));
msg = QIMDehide(y, delta, length(data));
m = reshape(msg, [64, 64]);
sn = Similar(d,m);
subplot(1,3,2);imshow(m);title(sprintf('delta:%d, similar: %.2f',delta, sn));