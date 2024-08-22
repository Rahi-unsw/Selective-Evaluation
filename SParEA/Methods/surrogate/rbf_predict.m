function y = rbf_predict(X, rbmodel)
% RBFPREDICT - To predict using radial basis function model
%
% Call
%    y = rbf_predict(X, rbmodel)
%
% Input
% X      : Data Points
% rmodel : Radial Basis model obtained using rbf_model
%
% Output:
% y   : Predicted response
%

	% Check arguments
	if nargin ~= 2
		error('rbf_predict requires 2 input arguments')
	end

	y = sim(rbmodel, X')';
end
