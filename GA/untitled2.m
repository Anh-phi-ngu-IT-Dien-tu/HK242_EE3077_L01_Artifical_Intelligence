clc
clear all

X1=[0 1 0 0];
X2=[0 0 1 0];
X3=[0 0 1 0];
X4=[0 0 0 1];

X=[X1;X2;X3;X4];

T1=[0.03 0.13 1e-10 0.84];
T2=[0.25 0.2 0.05 0.5];
T3=[0.11 0.17 0.68 0.03];
T4=[0.11 0.02 0.08 0.79];
T=[T1;T2;T3;T4]

mse=0
for i=1:size(T,1)
    ce=sum(X(i,:).*log2(T(i,:)));
    mse=mse+ce
end
mse=-1/size(T,2)*mse

w=zeros(40,1)
disp(w(1:5,1))