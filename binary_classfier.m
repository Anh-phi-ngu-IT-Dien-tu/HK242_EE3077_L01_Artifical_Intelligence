% Load dữ liệu
load('fruit_data.mat');

% Kiểm tra kích thước dữ liệu
disp("Kích thước của features:");
disp(size(features));

disp("Các nhãn phân loại:");
disp(unique(class_labels));

% Chia dữ liệu thành training (80%) và test (20%)
cv = cvpartition(length(class_labels), 'HoldOut', 0.1);
X_train = features(training(cv), :);
y_train = class_labels(training(cv));
X_test = features(test(cv), :);
y_test = class_labels(test(cv));

disp("Kích thước tập huấn luyện:");
disp(size(X_train));
disp("Kích thước tập kiểm tra:");
disp(size(X_test));

% Huấn luyện mô hình SVM đa lớp
svm_model = fitcecoc(X_train, y_train);

% Dự đoán trên tập kiểm tra
y_pred = predict(svm_model, X_test);

% Đánh giá mô hình
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Độ chính xác của mô hình SVM: %.2f%%\n', accuracy * 100);

save('svm_fruit_classifier.mat', 'svm_model');
