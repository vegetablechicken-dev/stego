A = [0 1 0 0; 0 0 1 0; 0 0 0 1; -125 -5 -20 -0.45];
B = [0; 0; 0; 100];
C = [1 0 0 0];
D = 0;

t_span = [0 2];
u = 1;
x0 = [0 0 0 0];


[t, x] = ode45(@(t, x) A*x+B*u, t_span, x0);
y = C * x' + D * u;

figure;
subplot(2, 1, 1);
plot(t, x);
title("States of the System");
legend('x1', 'x2', 'x3', 'x4');
xlabel("Time(s)");
ylabel("State Variable x");

subplot(2, 1, 2);
plot(t, y);
title("Output y of the System");
xlabel("Time(s)");
ylabel("Output y");

