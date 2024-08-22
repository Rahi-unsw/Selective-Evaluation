function [f,g] = g6(x)
if nargin == 0
    prob.nx = 2;
    prob.nf = 1;
    prob.ng = 2;
    prob.range(1,:) = [13,100] ;
    prob.range(2,:) = [0,100];
    f = prob;
else
    [f,g] = g6_true(x);
end
return


function [f,g] = g6_true(x)

g(:,1) = - ((x(:,1)-5).^2 + (x(:,2)-5).^2 - 100);
g(:,2) = -(-(x(:,1)-6).^2 - (x(:,2)-5).^2 + 82.81);
f(:,1) = (x(:,1)-10).^3 + (x(:,2)-20).^3;
return
