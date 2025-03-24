clc
clear all
%% Lấy dữ liệu
X1 = readmatrix('data.xlsx','Sheet','training','Range','C2:F2');
X2 = readmatrix('data.xlsx','Sheet','training','Range','C5:F5');
X3 = readmatrix('data.xlsx','Sheet','training','Range','C8:F8');

X4 = readmatrix('data.xlsx','Sheet','training','Range','C3:F3');
X5 = readmatrix('data.xlsx','Sheet','training','Range','C6:F6');
X6 = readmatrix('data.xlsx','Sheet','training','Range','C9:F9');

X7 = readmatrix('data.xlsx','Sheet','training','Range','C4:F4');
X8 = readmatrix('data.xlsx','Sheet','training','Range','C7:F7');
X9 = readmatrix('data.xlsx','Sheet','training','Range','C10:F10');

X=[X1 X2 X3;X4 X5 X6;X7 X8 X9]';
Y1=strings(4,1);
Y2=strings(4,1);
Y3=strings(4,1);
for i=1:4
    Y1(i,1)="quả quýt";
    Y2(i,1)="quả ổi";
    Y3(i,1)="quả táo xanh";
end

Y=[Y1;Y2;Y3];
%% Tạo bộ classifier 'NumNeighbors':k 'Standardize',1(Chuẩn hóa) ,khoảng cách sử dụng: mặc định là 'euclidean'
Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);

%% Sử dụng classifier 
X=[1 1 7]
label = predict(Mdl,X)
