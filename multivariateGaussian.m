function p = multivariateGaussian(X, mu, Sigma2,returnLog)
%MULTIVARIATEGAUSSIAN Computes the probability density function of the
%multivariate gaussian distribution.
%    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
%    density function of the examples X under the multivariate gaussian 
%    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
%    treated as the covariance matrix. If Sigma2 is a vector, it is treated
%    as the \sigma^2 values of the variances in each dimension (a diagonal
%    covariance matrix)
%	X is a Nxd matrix of N data instances each of d dimensions
%	mu is a 1xd d vector. Which is the mean vector of the gaussian of dimension d. 
%   returnLog should be 1 if the log of the probability should be returned. 0 if
%   not

k = length(mu);

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
	%make into diagonal covariance matrix if Sigma2 is a vector
    Sigma2 = diag(Sigma2);	
end

%for each training example, subtract from feature i (in column i) the mean of feature i. i.e. mean normalize each column
X = bsxfun(@minus, X, mu(:)'); 

%Compute probabilities from multi-variate gaussian equation
if (returnLog==0)
	p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));
else
	p = -log((2 * pi) ^ (k / 2) * det(Sigma2) ^ (0.5)) + (-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));
end

end