clc
clear all

I1=[1 0 0 0];
I2=[0 1 0 0];
I3=[0 0 1 0];
I4=[0 0 1 0];
I=[I1;I2;I3;I4];

X1=[0 1 0 0];
X2=[0 0 1 0];
X3=[0 0 1 0];
X4=[0 0 0 1];

X=[X1;X2;X3;X4];

T1=[0.03 0.13 1e-10 0.84];
T2=[0.25 0.2 0.05 0.5];
T3=[0.11 0.17 0.68 0.03];
T4=[0.11 0.02 0.08 0.79];
T=[T1;T2;T3;T4];
nvars=40

rng default % For reproducibility
FitnessFunction = @(w)cross_entropy_fitness(w,X,T)

[best_wb, best_loss] = ga(FitnessFunction, nvars);


function ce = cross_entropy_fitness(w,X, T)
    % Cross-entropy loss
    ce=0;
    for i=1:size(T,1)
        temp=-1/size(T,2)*sum(X(i,:).*log2(T(i,:)));
        ce=ce+temp;
    end
end