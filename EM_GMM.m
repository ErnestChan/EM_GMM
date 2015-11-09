function P = EM_GMM(data, initialP, maxIter,needDiag,printLikelihood)
% Params: data - a NxP matrix where the rows are points and the columns are
%               variables. e.g. N 2-D points would have N rows and 2
%               columns
%       initialP - a struct of the initial parameters for the GMM.
%       maxIter - the number of iterations to run expectation-maximization
%               (EM) for fitting the GMMs
%       needDiag - set as 1 if you want diagonal covariance matrices for
%               the components. Set as 0 otherwise.
%       printLikelihood - set as 1 to print the log likelihood at each EM
%                   iteration
% Returns: P - a struct that holds the parameters of the GMM. P.comp is a
%           list of structs that holds the parameters for each component.
%           For component i, P.comp(i) contains that component's mixing
%           weight "alpha", and the mean "mu" and covariance matrix
%           "sigma2" of the multivariate Gaussian
% I used MAP estimation to estimate the GMM parameters since using just MLE 
% resulted in many numerical issues. See Kevin Murphy's book for equations and a
% graph of MLE & MAP numerical issues vs. data dimensionality.

P = initialP;
M = length(P.comp); %M is the number of Gaussian components
[N, D] = size(data);
loglikelihood = zeros(maxIter,1);

%prior for MAP estimation to v0.
v0 = 2*(D+2);

for iter = 1:maxIter
% E-STEP. compute component probabilities for each data instance
% We are given the params for each component
	componentProbs = zeros(N,M);
	for m = 1:M
		%compute componentProbs for all data instances in log space
		componentProbs(:,m) = log(P.comp(m).alpha)+multivariateGaussian(data, P.comp(m).mu, P.comp(m).sigma2,1);
	end
	%logsumexp across each row of componentProbs to get the probability of each data instance. This is will also
	%be the normalization term for component probs later.
	pX = logsumexp(componentProbs);
	if (sum(isnan(pX(:)))>0)
		val='nan in pX' %print out that there is a nan
		return;
	end
	loglikelihood(iter) = sum(pX)/N;

	if (printLikelihood==1)
		%Print out log likelihood and iteration number
		disp(sprintf('EM_GMM iteration %d: log likelihood: %f', iter, loglikelihood(iter)));
  	end

	%normalize component probs in log space, then exponentiate to get back
	%regular probability space
	componentProbs = exp(bsxfun(@minus,componentProbs,pX));

	nk = sum(componentProbs);
	
% M-STEP. Compute the params for the GMM
% alpha - the component probabilities
% mu - the mean vector of each component gaussian
% sigma2 - the covariance matrix of each component gaussian
	for k=1:M
        %compute new alpha and the new mean vector based on the E-step
        %results
		P.comp(k).alpha = nk(k)/N;
		P.comp(k).mu = (1/nk(k))*sum(bsxfun(@times,componentProbs(:,k),data))';
		%Calculate full covariance matrix if needDiag==0. Compute diagonal covariance matrix if it equals 1
		if (needDiag==0)
            %MAP estimation of params. Done to avoid numerical issues with covariance matrices with very
            %small numbers. Priors are S0 and v0. Sk is basically the MLE calc for sigma2 without the
            % 1/nk term. Equations in Murphy pg. 357
            X = bsxfun(@minus,data,P.comp(k).mu');
            S0 = diag((sum(X.^2)/N))/(M^(1/D));	
            Sk = bsxfun(@times,componentProbs(:,k),X)'*X;	
            P.comp(k).sigma2 = (S0 + Sk)/(v0 +nk(k));
		else
			% compute diagonal covariance matrix. Main difference is we .* data with itself (which is like squaring)
			% instead of multiplying data' by data.
            X = bsxfun(@minus,data,P.comp(k).mu');
            S0 = diag((sum(X.^2)/N))/(M^(1/D));	
            %kept 1/nk(k) term in Sk since taking it out decreased performance on basic tasks
            Sk = diag((1/nk(k))*(componentProbs(:,k)'*(data.*data)) - (P.comp(k).mu.^2)');
            P.comp(k).sigma2 = (S0 + Sk)/(v0 +nk(k));
        
		end
	end

end

end

