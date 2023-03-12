x = linspace(-5,5);
y = sigmd(x);

plot(x,y);
xlabel('x','FontSize',13)
ylabel('\sigma(x)','FontSize',13)
grid on

function result = sigmd(x)
    result = 2/pi*atan(1.4*pi*x/2);
end