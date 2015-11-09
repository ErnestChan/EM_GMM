function cost=clusteringCost(X,IDX,Centers)
%clusteringCost computes the cost of kmeans clustering
% Params: X - a NxP matrix where the rows are points and the columns are
%               variables. e.g. N 2-D points would have N rows and 2
%               columns
%         IDX - a 1xP vector that specifies the cluster each point in X is
%               assigned to.
%         Centers - a CxP matrix of cluster centers where C are the number
%               of clusters
m=size(X,1);
cost=0;
for i=1:size(Centers,1)
	%Find the difference between each point and its cluster center. Normalize each row
    normDifferences = sum(bsxfun(@minus, X(IDX==i,:), Centers(i,:)).^2,2);
	% X'X is the same as squaring each element in X and summing the vector. Here we square each
	% element in normDifferences and sum them up, then divide by m. This is the cost for cluster i.
    cost=cost+(1/m)*sum(normDifferences);
end


end