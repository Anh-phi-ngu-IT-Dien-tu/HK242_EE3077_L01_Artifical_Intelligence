clc
clear all

I1=[1 0 0 0];
I2=[0 1 0 0];
I3=[0 0 1 0];%Đầu vào là one -hot vector của bộ chữ 'h','e','l','l'
I4=[0 0 1 0];
I=[I1;I2;I3;I4];

X1=[0 1 0 0];
X2=[0 0 1 0];
X3=[0 0 1 0];
X4=[0 0 0 1];

X=[X1;X2;X3;X4];

T1=[0.03 0.13 1e-10 0.84];
T2=[0.25 0.2 0.05 0.5];
T3=[0.11 0.17 0.68 0.03];%Đầu ra là softmax , các giá trị =0 thì ta cho 
T4=[0.11 0.02 0.08 0.79];% giá trị gần bằng 0 để hàm log của cross entrophy không bị bằng 0
T=[T1;T2;T3;T4];

nvars=40
rng default % For reproducibility
FitnessFunction = @(w)cross_entropy_fitness(w,I,T)
opts = optimoptions('ga', 'MaxGenerations', 100, 'Display', 'iter');
[best_wb, best_loss] = ga(FitnessFunction, nvars,[], [], [], [], [], [], [], opts);
best_wb=best_wb'

Parameters=strings(40,1)
for i=1:12
    str="W_hx%d";
    Parameters(i,1)=compose(str,i);
end
for i=1:9
    str="W_hh%d";
    Parameters(i+12,1)=compose(str,i);
end
for i=1:12
    str="W_hy%d";
    Parameters(i+21,1)=compose(str,i);
end
for i=1:3
    str="b_h%d";
    Parameters(i+33,1)=compose(str,i);
end
for i=1:4
    str="b_y%d";
    Parameters(i+36,1)=compose(str,i);
end

Weights=string(best_wb);
Table=table(Parameters,Weights);
filename = 'weight_after_training.xlsx';
writetable(Table,filename,'Sheet','weight','Range','A1')

function loss = cross_entropy_fitness(w,X, T)%định nghĩa hàm loss theo w
    h = zeros(3,1);
    loss = 0;
    W_xh = reshape(w(1:12),3,4); 
    W_hh = reshape(w(13:21),3,3); 
    W_hy = reshape(w(22:33),4,3); 
    b_h = reshape(w(34:36), 3, 1); 
    b_y = reshape(w(37:40), 4, 1);
    
    for t = 1:4% ngõ vào dự đoán so với ngõ vào mục tiêu 
        h = tanh(W_xh*X(t,:)' + W_hh*h + b_h);
        y = softmax(W_hy*h + b_y);
        target=T(t,:)';
        loss = loss - sum(target .* log(y));  % tránh log(0)
    end
    
end