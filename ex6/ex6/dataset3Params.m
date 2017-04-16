function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [0.01 0.03 0.1 0.3 1 3 10 30];
%steps = [0.01 0.03];
lsteps = length(steps);
index = 1;
% error for each 
errors_train = zeros(lsteps * lsteps, 3);
errors_val = zeros(lsteps * lsteps, 3);

for i = 1 : lsteps
  for j = 1 : lsteps
    % Try every model combination for C and sigma
    fprintf('Trying different models...\n\n');
    fprintf('Model No. %d\n', index);
    C = steps(i);
    sigma = steps(j);
    fprintf('C is %d\n', C);
    fprintf('sigma is %d\n', sigma);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions_train = svmPredict(model, X);
    predictions_val = svmPredict(model, Xval);
    error_train = mean(double(predictions_train ~= y));
    error_val = mean(double(predictions_val ~= yval));
    errors_train(index, :) = [ error_train , C, sigma ];
    errors_val(index, :) = [ error_val, C, sigma ];
    index += 1;
  end
end

[~, I] = min(errors_val);

error_val = errors_val(I(1), 1);
C = errors_val(I(1), 2);
sigma = errors_val(I(1), 3);

fprintf('Estimated best parameters C = %f and sigma = %f\n', C, sigma);

plot(errors_train(:,2), errors_train(:,1), errors_val(:,2), errors_val(:,1));
legend('Train', 'Cross Validation');
xlabel('C and sigma');
ylabel('Error');

fprintf(' C\t\tsigma\t\tTrain Error\tValidation Error\n');
for i = 1:length(errors_val)
	fprintf(' %f\t%f\t%f\t%f\n', ...
            errors_val(i,2), errors_val(i,3) , errors_train(i,1), errors_val(i,1));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

% =========================================================================

end
