function rbmodel = rbf_train(X, Y)
% RBFMODEL - To construct radial basis function network model
%
% Call
%    rbmodel = rbf_train(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
% 
% Output
% rbmodel : RBF Model
%

	% Check arguments
	if nargin ~= 2
		error('rbf_model requires 2 input arguments')
	end

	% Check design points
	[m1 nx] = size(X);
	[m2 ny] = size(Y);
	if m1 ~= m2
		error('X and Y must have the same number of rows')
	end

	rbmodel = newrbe(X', Y');
end
