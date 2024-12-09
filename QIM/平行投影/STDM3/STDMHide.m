function o = STDMHide(I, data, delta)

block = [8, 8];
si = size(I);
lend = length(data);
N = floor(si(2) / block(2));%there are N blocks in each row
M = min(floor(si(1) / block(1)), ceil(lend / N));%there are M block in each colomn
if lend < M*N
    data = [data zeros(1, M*N -lend)];
end

o = I;
idx = 1;


for i = 0 : M - 1
    
    rst = i * block(1) + 1;
    red = (i + 1) * block(1);

    for j = 0 : N - 1
        
        cst = j * block(2) + 1;
        ced = (j + 1) * block(2);
        tmp = I(rst:red, cst:ced);
        tmp = dct2(double(tmp));
       
        x = zeros(1,block(1));
        for k = block(1):-1:1
            m = block(1)+1 - k;%makes position for diagnal coiefficients           
            x(m) = tmp(k, m);
        end;
        
        p = norm(x);
        v = x / p;
        p = Quantificate(p, data(idx), delta);
        if sign(p) == -1
            p = p + delta;
        end
        x = p * v;
                
        for k = block(1):-1:1
            m = block(1)+1 - k;%makes position for diagnal coiefficients           
            tmp(k, m) = x(m);
        end;
        
        o(rst:red, cst:ced) = idct2(tmp);
        idx = idx + 1;

    end;
end;
