function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, C, sigma)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ---------------------- Sample Solution ----------------------

for i = 1:m
  model = svmTrain(X(1:i,:), y(1:i), C, @(x1, x2) gaussianKernel(x1, x2, ...
    sigma));
  predictions = svmPredict(model, X);
  valPredictions = svmPredict(model, Xval);
  [error_train(i), ~] = mean(double(predictions ~= y);
  [error_val(i), ~] = mean(double(valPredictions ~= yval);
end

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for support vector machine with gaussian kernel')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

% -------------------------------------------------------------

% =========================================================================

end
