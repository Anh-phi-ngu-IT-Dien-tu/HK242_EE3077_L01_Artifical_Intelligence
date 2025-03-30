#include <iostream>
#include "q_learning.h"
#include "Eigen"
#include <fstream>
#include <string>

int main() 
{
    //declare reward matrix
    Eigen::Matrix<double,6,6> R;
    R.setZero();
    //get reward matrix from csv file
    std::ifstream file("data_input.csv");
    std::string line;
    int i=0,j=0;
    while(std::getline(file,line))
    {
        j=0;
        std::stringstream ss(line);
        std::string value;
        
        while (getline(ss, value,',')) 
        {
            R(i,j)=std::stod(value);
            j=j+1;
        }
        i++;
    }
    file.close();

    //declare everything for Q learning algorithm
    Q_Learning path_planning(6,6,0.8,0.2);
    //load reward matrix
    path_planning.setReward(R);
    //start learning algorithm
    path_planning.QLearningAgorithm();
    //print the learning result
    std::cout<<path_planning;
    //store Q result into csv file
    std::ofstream outfile("Q_Matrix_After_Training.csv");
    Eigen::IOFormat HeavyFmt(Eigen::StreamPrecision, 0, ",", "\n");
    outfile<<path_planning.getQ().format(HeavyFmt);
    outfile.close();
    system("pause");
    return 0;    
}

