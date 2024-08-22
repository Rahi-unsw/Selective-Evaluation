function [f,g] = helical_spring(x)
if nargin == 0
    prob.nx = 3;
    prob.nf = 1;
    prob.ng = 9;
    prob.range(1,:) = [0.05, 2.0];
    prob.range(2,:) = [0.25, 1.3];
    prob.range(3,:) = [2.0, 20.0];
    f = prob;
else
    [f,g] = helical_spring_true(x);
end
return


%% Design of tension/compression spring
% Minimize weight subject to constraints on minimum deflection, shear stress
% surge frequency
% design variables - the mean coil diameter, wire diameter, number of coils
function [f,g] = helical_spring_true(x)

D = x(:,1);
d = x(:,2);
N = round(x(:,3));
N(N<2)=2;
N(N>20)=20;

pi = 3.14159;

S = 189000;				% psi
E = 3.0e7;				% psi
G = 1.15e7;				% psi
F_max = 1000; 			% lb
l_max = 14;				% in
d_min = 0.2;			% in
D_max = 3.0;			% in
F_p = 300;				% lb
delta_pm = 6;			% in
delta_w = 1.25;			% in
C_E = 1.0;

C = D./d;
C_f = ((4.*C-1)./(4.*C-4)) + 0.615./C;
K = (G.*d.^4) ./ (8.*N.*D.^3);
delta = F_max ./ K;
l_f = delta + 1.05.*((N+2).*d);

delta_p = F_p ./ K;
alpha0 = l_f .* K;
term = 1 - (2.*pi.^2.*(1-G./E))./(1+2.*G./E).*(C_E.*D./l_f).^2;
term(term<0)=0;
C_B = 1./(2.*(1-G./E)) .* (1 - sqrt(term));

P_crit = alpha0 .* C_B;

g(:,1) = -(S - (8 .* C_f .* F_max .* D)./(pi .* d.^3));
g(:,2) = -(l_max - l_f);
g(:,3) = -(d - d_min);
g(:,4) = -(D_max - (D+d));
g(:,5) = -(C - 3);
g(:,6) = -(delta_pm - delta_p);
g(:,7) = -(l_f - delta_p - (F_max-F_p)./K - 1.05.*((N+2).*d));
g(:,8) = -((F_max-F_p)./K - delta_w);
g(:,9) = -(P_crit./1.25 - F_max);

f(:,1) = pi.^2./4 * ((N+2).*D.*d.^2);

return
