function z=Similar(x,y)
len=min(length(x(:)),length(y(:)));
x=double(x(1:len));
y=double(y(1:len));
z=sum(x.*y)/(sqrt(sum(x.^2))*sqrt(sum(y.^2)));