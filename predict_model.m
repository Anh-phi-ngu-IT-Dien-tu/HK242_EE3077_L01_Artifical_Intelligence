% Load mô hình đã lưu
load('svm_fruit_classifier.mat', 'svm_model');

disp("Mô hình SVM đã được load thành công!");

% Dữ liệu test
test_sample = [6, 7, 6]; % Đơn vị cm

% Dự đoán loại quả
predicted_class = predict(svm_model, test_sample);

% Hiển thị kết quả

disp("Loại quả dự đoán:");
for i = 1:length(predicted_class)
    fprintf('Dự đoán %d: %s\n', i, predicted_class{i});
end
