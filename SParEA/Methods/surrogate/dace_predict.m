function y = dace_predict(X, dmodel)
% DACEPREDICT - To predict using kriging model
%
% Call
%    y = dace_predict(X, rmodel)
%
% Input
% X      : Data Points
% rmodel : Kriging model obtained using dace_model
%
% Output:
% y   : Predicted response
%

	% Check arguments
	if nargin ~= 2
		error('dace_predict requires 2 input arguments')
	end

	f = predictor(X, dmodel);
	if size(f,1) == size(X,1)
		y = f;
	else
		y = f';
	end
end
