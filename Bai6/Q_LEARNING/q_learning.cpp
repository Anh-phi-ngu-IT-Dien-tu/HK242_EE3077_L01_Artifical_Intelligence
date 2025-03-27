#include "q_learning.h"

Q_Learning::Q_Learning(unsigned int size_of_Q_and_R,unsigned int num_of_episode,double set_gamma,double set_learning_rate)
{
    Q.resize(size_of_Q_and_R,size_of_Q_and_R);
    R.resize(size_of_Q_and_R,size_of_Q_and_R);
    Q.setZero();
    R.setZero();
    size=size_of_Q_and_R;
    target_state=size-1;
    if(num_of_episode>0)
        episode=num_of_episode;
    else
        episode=5;    
    gamma=set_gamma;
    alpha=set_learning_rate;

}

Q_Learning::~Q_Learning()
{
    
}

void Q_Learning::resetQ()
{
    Q.setZero();
}

void Q_Learning::setZeroForR()
{
    R.setZero();
}

void Q_Learning::setReward(Eigen::MatrixXd Reward)
{
    if(Reward.rows()<=size && Reward.cols()<=size)
        R=Reward;
    else
        std::cout<<"Invalid number of columns and rows\nThe numbers of columns and rows have to be the same as the size we have defined"<<std::endl;

}

Eigen::MatrixXd Q_Learning::getQ()const
{
    return Q;
}

Eigen::MatrixXd Q_Learning::getR()const
{
    return R;
}

void Q_Learning::Fill_the_temp(Eigen::MatrixXd M,unsigned int row)
{
    if(row<size)
    {
        for(unsigned int i=0;i<size;i++)
        {
            temp.push_back(M(row,i));
        }
    }
    else
        std::cout<<"Mismatch index when proceding Fill_the_temp"<<std::endl;
}

void Q_Learning::Delete_the_temp()
{
    for(int i =temp.size();i>0;i--)
    {
        temp.pop_back();
    }
}

void Q_Learning::Delete_possible_action()
{
    for(int i=possible_action.size();i>0;i--)
    {
        possible_action.pop_back();
    }
}

void Q_Learning::seeAllPosibleAction()
{
    for(int i=0;i<possible_action.size();i++)
    {
        std::cout<<"possible action "<<i<<" = "<<possible_action[i];
    }
    std::cout<<"\n";
}

//max_element´returns an iterator 
//basically it is like a pointer that points to an element in a container
//usually use for vector
void Q_Learning::QLearningAgorithm()
{
    srand(time(0));
    for(int i=0;i<episode;i++)
    {
        //select a random initial state
        #if Debuggging==1
        std::cout<<"\n\nepisode "<<i<<std::endl;
        #endif

        current_state=rand()%(size-1);
        
        //if the goal has not been reached
        while(current_state!=target_state)
        {
            #if Debuggging==1
            std::cout<<"\ncurrent state ="<<current_state<<std::endl;
            #endif

            //Select (randomly) one among all possible actions for the current state.
            for(unsigned int j=0;j<size;j++)
            {
                if(R(current_state,j)>=0)
                {  
                    #if Debuggging ==1
                    std::cout<<"R("<<current_state<<","<<j<<")="<<R(current_state,j)<<" ";
                    #endif
                    possible_action.push_back(j);
                }            
            }

            #if Debuggging ==1
            std::cout<<"\n";
            seeAllPosibleAction();
            #endif

            //Using this possible action, consider going to the next state.
            chose_action=rand()%possible_action.size();

            #if Debuggging==1
            std::cout<<"possible_action ="<<possible_action[chose_action]<<std::endl;
            #endif

            //get maximum Q value for next state based on all possible action
            Fill_the_temp(Q,possible_action[chose_action]);
            double max=*std::max_element(temp.begin(), temp.end());
            Delete_the_temp();

            #if Debuggging==1
            std::cout<<"Max Q in next state ="<<max<<std::endl;
            #endif

            //compute 
            Q(current_state,possible_action[chose_action])=(1-alpha)*Q(current_state,possible_action[chose_action])+alpha*(R(current_state,possible_action[chose_action])+gamma*max);
            #if Debuggging==1
            std::cout<<"Q("<<current_state<<","<<possible_action[chose_action]<<")="<<Q(current_state,possible_action[chose_action])<<std::endl;
            #endif

            //Set the next state as the current state.
            current_state=possible_action[chose_action];
            Delete_possible_action();
        }
    }
}

std::ostream &operator<<(std::ostream &output,const Q_Learning &q)
{
    return output<<"Q=\n"<<q.getQ()<<"\nR=\n"<<q.getR()<<std::endl;
}

