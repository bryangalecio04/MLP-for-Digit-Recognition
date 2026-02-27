% Obtain dataset data for training/validation
training_validation_path = fullfile(pwd, "../dataset/training_validation/*.tif");
training_validation_dir = dir(training_validation_path);
num_images = length(training_validation_dir);

digit_input = zeros(num_images,256); % Images are 16x16, so flattened will be 1x256
digit_label = zeros(num_images, 10); % Labels for images, 100 images, 100 labels, 10 classes

for i = 1:num_images
    digit_image = imread(fullfile(training_validation_dir(i).folder, training_validation_dir(i).name));
    digit_image = double(digit_image(:)'); % Reshaping from 16x16 to 1x256

    digit_input(i, :) = digit_image; % Store the flattened image in the input matrix

    label = str2double(training_validation_dir(i).name(5)) + 1;
    digit_label(i, label) = 1;
end

save("train_network.mat", "net", "digit_input", "digit_label")

% Uncomment post training to save the trained network to a .mat file for later use in inference
% save("network.mat", "network");