function visualize2DGaussians(data,P)
% visualizeAll visualizes the data, and the gaussians fit to the data. It
% just visualizes the data if the second parameter is not given
% Inputs: P - a struct that holds the parameters of the GMM. P.comp is a
%           list of structs that holds the parameters for each component.
%           For component i, P.comp(i) contains that component's mixing
%           weight "alpha", and the mean "mu" and covariance matrix
%           "sigma2" of the multivariate Gaussian

hold off
%Plot  data
if (nargin==1)
	plot(data(:, 1), data(:, 2), 'bdata');
	hold on;
end
%compute range of x and y axes
minX = min(data(:,1)); maxX = max(data(:,1));
minY = min(data(:,2)); maxY = max(data(:,2));
xBuffer = (maxX-minX)/5;
yBuffer = (maxY-minY)/5;
axesRange = [minX-xBuffer,maxX+xBuffer,minY-yBuffer,maxY+yBuffer];
%if more than one input argument, then GMM params are given.
if (nargin > 1)
	for k=1:length(P.comp)
		visualizeFit(data, P.comp(k).mu, P.comp(k).sigma2,axesRange);
		hold on;
	end
end
axis(axesRange);
xlabel('x');
ylabel('y');
grid on
hold off
end