function rmodel = rsm1_train(X, Y)
% RSMMODEL - To construct quadratic response surface model
%
% Call
%    rmodel = rsm_model(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
% 
% Output
% rmodel : RSM Model - a struct with elements
%

	% Check arguments
	if nargin ~= 2
		error('rsm_model requires 2 input arguments')
	end

	% Check design points
	[m1 nx] = size(X);
	[m2 ny] = size(Y);
	if m1 ~= m2
		error('X and Y must have the same number of rows')
	end

	% Construct F matrix for quadratic model
	F = regpoly1(X);
	nf = size(F, 2);

	% Regression model
	b = zeros(nf, ny);
	for i = 1:ny
		F = regpoly1(X);
		b(:,i) = regress(Y(:,i), F);
	end

	rmodel = struct('nx', nx, 'ny', ny, 'b', b);
end
