function dmodel = dace_train(X, Y)
% DACEMODEL - To construct kriging model
%
% Call
%    rmodel = dace_model(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
% 
% Output
% dmodel : DACE Model
%

	% Check arguments
	if nargin ~= 2
		error('dace_model requires 2 input arguments')
	end

	% Check design points
	[m1 nx] = size(X);
	[m2 ny] = size(Y);
	if m1 ~= m2
		error('X and Y must have the same number of rows')
	end

	theta = 1*ones(1, nx);
	range = minmax(X');

	[dmodel, perf] = dacefit(X, Y, @regpoly0, @corrgauss, theta, max(1e-6,range(:,1)), range(:,2));
end
