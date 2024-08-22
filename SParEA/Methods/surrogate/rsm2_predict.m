function y = rsm2_predict(X, rmodel)
% RSMPREDICT - To predict using quadratic response surface model
%
% Call
%    y = rsm_predict(X, rmodel)
%
% Input
% X      : Data Points
% rmodel : Response surface model obtained using rsm_model
%
% Output:
% y   : Predicted response
%

	% Check arguments
	if nargin ~= 2
		error('rsm_predict requires 2 input arguments')
	end

	% Check data points
	[m nx] = size(X);
	if nx ~= rmodel.nx
		error('X must be consistent with rmodel')
	end

	% Construct F matrix for prediction
	F = regpoly2(X);
	y = F * rmodel.b;
end
