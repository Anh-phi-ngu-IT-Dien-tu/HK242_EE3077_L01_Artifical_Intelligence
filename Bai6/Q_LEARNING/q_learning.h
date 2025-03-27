#include <iostream>
#include <Eigen>
#include <vector>
#include <ctime>
#include <algorithm>

/**
 * Q(state,action)=Q(current_state,possible_action[chose_action])
 * 
**/
#define Debuggging 1
class Q_Learning
{
    private:
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    unsigned int size;
    std::vector<int> possible_action;
    int current_state,episode;
    double gamma,alpha;
    std::vector<double> temp;
    int chose_action,target_state;

    void Fill_the_temp(Eigen::MatrixXd M,unsigned int row);
    void Delete_the_temp();
    void Delete_possible_action();
    public:
    Q_Learning(unsigned int size_of_Q_and_R,unsigned int num_of_episode,double set_gamma,double set_learning_rate);
    ~Q_Learning();

    //getter
    Eigen::MatrixXd getQ()const;
    Eigen::MatrixXd getR()const;

    void resetQ();
    void setZeroForR();
    void setReward(Eigen::MatrixXd Reward);
    void QLearningAgorithm();
    void seeAllPosibleAction();
};
 

std::ostream &operator<<(std::ostream &output,const Q_Learning &q);