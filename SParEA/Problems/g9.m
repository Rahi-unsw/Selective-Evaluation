function [f,g] = g9(x)
if nargin == 0
    prob.nx = 7;
    prob.nf = 1;
    prob.ng = 4;
    for i = 1:prob.nx
        prob.range(i,:) = [-10,10];
    end
    f = prob;
else
    [f,g] = g9_true(x);
end
return


function [f,g] = g9_true(x)

g(:,1) = -(127 - 2*x(:,1).^2 - 3*x(:,2).^4 - x(:,3) - 4*x(:,4).^2 - 5*x(:,5));
g(:,2) = -(282 - 7*x(:,1) - 3*x(:,2) - 10*x(:,3).^2 - x(:,4) + x(:,5));
g(:,3) = -(196 - 23*x(:,1) - x(:,2).^2 - 6*x(:,6).^2 + 8*x(:,7));
g(:,4) = -(-4*x(:,1).^2 - x(:,2).^2 + 3*x(:,1).*x(:,2) - 2*x(:,3).^2 - 5*x(:,6) + 11*x(:,7));
f(:,1) = (x(:,1)-10).^2 + 5*(x(:,2)-12).^2 + x(:,3).^4 + 3*(x(:,4)-11).^2 + 10*x(:,5).^6 ...
            + 7*x(:,6).^2 + x(:,7).^4 - 4*x(:,6).*x(:,7) - 10*x(:,6) - 8*x(:,7);
return
