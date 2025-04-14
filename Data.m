% Tạo dữ liệu theo đúng bảng trong hình
class_labels = ["quả quýt"; "quả quýt"; "quả quýt"; "quả quýt";...
                "quả ổi"; "quả ổi"; "quả ổi"; "quả ổi"; ...
                "quả táo xanh"; "quả táo xanh"; "quả táo xanh";"quả táo xanh"];

object = ["object1"; "object2";"object3";"object4";...
          "object1"; "object2";"object3";"object4";... 
          "object1"; "object2";"object3";"object4"];

DuongKinhTren = [5;4;6;5.7;...
                 7;6.7;6;5;...
                 3.4;3;3.3;2.8];
DuongKinhDuoi = [6;4.7;6.3;6.1;...
                 8;7.5;6.5;5.7;...
                 1;1.5;1.8;1.6];
ChieuCao = [5.8;5;5.4;5.2;...
            6.5;6.2;5.6;6.3;...
            4.5;4.7;4.4;4.3];

% Tạo bảng dữ liệu
data = table(class_labels, object, DuongKinhTren, DuongKinhDuoi, ChieuCao);

% Hiển thị dữ liệu
disp(data);

% Chuyển nhãn class thành số
class_numeric = zeros(size(class_labels));
class_numeric(strcmp(class_labels, "quả quýt")) = 1;  % Gán "quả quýt" là 1
class_numeric(strcmp(class_labels, "quả ổi")) = 2;    % Gán "quả ổi" là 2
class_numeric(strcmp(class_labels, "quả táo xanh")) = 3; % Gán "quả táo xanh" là 3

features = [DuongKinhTren, DuongKinhDuoi, ChieuCao];
% Hiển thị dữ liệu dạng số
disp("Dữ liệu số hóa:");
disp([class_numeric, features]);

save('fruit_data.mat', 'data', 'class_numeric', 'features');

