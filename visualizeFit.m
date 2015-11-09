function visualizeFit(X, mu, sigma2, axesRange)
%VISUALIZEFIT Visualize the dataset and its estimated distribution.
%   VISUALIZEFIT(X, p, mu, sigma2,axesRange) This visualization shows you the 
%   probability density function of the Gaussian distribution. Each point
%   has a location (x1, x2) that depends on its feature values.
%   Input paramater - axesRange gives the range of the axes of the figure
%   to be ploted on. It determines the range of the meshgrid created for
%   the contours.

% Find min and max values for the meshgrid
meshMin = min(axesRange(1),axesRange(3));
meshMax = max(axesRange(2),axesRange(4));
%Create meshgrid.
[X1,X2] = meshgrid(meshMin:0.5:meshMax); 
Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2,0);
Z = reshape(Z,size(X1));

plot(X(:, 1), X(:, 2),'bx');
hold on;
% Do not plot if there are infinities
if (sum(isinf(Z)) == 0)
    contour(X1, X2, Z, 10.^(-9:3:-2)');
end
hold off;

end