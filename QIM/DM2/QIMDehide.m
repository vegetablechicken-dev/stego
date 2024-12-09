function o = QIMDehide(I, delta, len)

block = [8, 8];
si = size(I);
N = floor(si(2) / block(2));%there are N blocks in each row
M = floor(si(1) / block(1));%there are M block in each colomn
o = zeros(1, M*N);
idx = 1;

for i = 0 : M - 1
    
    
    rst = i * block(1) + 1;
    red = (i + 1) * block(1);

    for j = 0 : N - 1
              
        cst = j * block(2) + 1;
        ced = (j + 1) * block(2);
        tmp = I(rst:red, cst:ced);
        tmp = dct2(double(tmp));
        
        to = zeros(1,block(1));
        for k = block(1):-1:1  
            l = block(1) + 1 - k;%makes position for diagnal coiefficients
            q00 = Quantificate(tmp(k, l), 0, delta);
            q10 = Quantificate(tmp(k, l), 1, delta);
            
            [~, pos] = min(abs(tmp(k,l) - [q00, q10]));
            to(l) = pos - 1;
        end;
        if sum(to) >= 4
            o(idx) = 1;
        else
            o(idx) = 0;
        end
         idx = idx + 1;
    end;
end;

o = o(1:len);