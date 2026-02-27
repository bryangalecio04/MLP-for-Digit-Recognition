results = network.TrainingResults;
net = network.Network;
input_weights = net.IW{1}; % Weights for Input Layer -> Hidden Layer
bias_1 = net.b{1}; % Hidden Layer Biases
hidden_layer_weights = net.LW{2,1}; % Weights for Hidden Layer -> Output Layer
bias_2 = net.b{2}; % Output Layer Biases

% Obtain dataset data for inference
testing_path = fullfile(pwd, "../dataset/testing/*.tif");
testing_dir = dir(testing_path);
num_images = length(testing_dir);

digit_input = zeros(num_images,256); % Images are 16x16, so flattened will be 1x256
target = zeros(num_images, 10); % Target for ANN, 50 images, 50 labels, 10 classes

for i = 1:num_images
    image = imread(fullfile(testing_dir(i).folder, testing_dir(i).name));
    image = double(image(:)'); % Reshaping from 16x16 to 1x256

    digit_input(i, :) = image; % Store the flattened image in the input matrix

    label = str2double(testing_dir(i).name(5)) + 1;
    target(i, label) = 1;
end

% Apply same preprocessing used during training to ensure consistent input distribution
digit_input = removeconstantrows('apply', digit_input', net.inputs{1}.processSettings{1});
digit_input = mapminmax('apply', digit_input, net.inputs{1}.processSettings{2});

% Tansig Activation Function
function out = tansig(x)
    out = tanh(x);
end

% Softmax Activation Function
function out = softmax(x)
    e = exp(x - max(x, [], 1));
    out = e ./ sum(e, 1);
end

% Input Layer -> Hidden Layer
hidden = tansig((input_weights * digit_input) + bias_1); % (neurons in hidden layer)x(observations)

% Hidden Layer -> Output
output = softmax((hidden_layer_weights * hidden) + bias_2); % (classes)x(observations)
output = (output)'; % Transpose to 50x10 (observations)x(classes)

% Error Rate
[~, predicted] = max(output, [], 2); % Index of highest probability class (observations)x1
[~, true_labels] = max(target, [], 2); % Index of true class from target (observations)x1

correct = sum(predicted == true_labels);
accuracy = correct / num_images * 100;
error_rate = 100 - accuracy;

fprintf('Error Rate: %.2f%%\n', error_rate);