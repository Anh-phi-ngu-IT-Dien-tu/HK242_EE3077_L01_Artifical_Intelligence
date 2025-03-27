#include <iostream>
#include "q_learning.h"
#include "Eigen"

int main() 
{
    Q_Learning path_planning(6,6,0.8,0.2);
    Eigen::Matrix<double,6,6> R{
        {-1,-1,-1,-1,0,-1},
        {-1,-1,-1,0,-1,100},
        {-1,-1,-1,0,-1,-1},
        {-1,0,0,-1,0,-1},
        {0,-1,-1,0,-1,100},
        {-1,0,-1,-1,0,100}
    };
    path_planning.setReward(R);
    path_planning.QLearningAgorithm();
    std::cout<<path_planning;
    return 0;    
}

