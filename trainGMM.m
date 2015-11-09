function P = trainGMM(data,numComponents,maxIter,needDiag,printLikelihood)
% Params: data - a NxP matrix where the rows are points and the columns are
%               variables. e.g. N 2-D points would have N rows and 2
%               columns
%       numComponents - the number of gaussian mixture components
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

%create initial parameters in initialP, which contain the alphas, means, and 
%covariance matrices of the GMM
minCost = inf;
kmeanRuns = 5;
i=0;
%Run kmeans multiple times, choose the assignments and centers with the lowest 
%clustering cost.
%Use try catch since kmeans will sometimes create an empty cluster
while(i<=kmeanRuns)
	errors = 0;
	try
		[IDX, Centers] = kmeans(data, numComponents);
	catch
		errors = errors+1;
	end
	if (errors==0)
		cost=clusteringCost(data,IDX,Centers);
		i=i+1;
		if (cost<minCost)
			minCost = cost;
			bestCenters = Centers;
		end
	end
end

%Initialize the mu's to be to be the cluster centers, the alpha's to be uniform, and 
%the covariance matrices to be identity matrices
for k = 1:numComponents
	initialP.comp(k).alpha = 1/numComponents;
	initialP.comp(k).mu = bestCenters(k,:)';
	initialP.comp(k).sigma2 = diag(ones(1,size(data,2)));
end

%Run EM to fit the GMMs using the initialized parameters
P = EM_GMM(data, initialP, maxIter,needDiag,printLikelihood);

end