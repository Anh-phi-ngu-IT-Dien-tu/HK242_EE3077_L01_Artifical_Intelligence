import numpy as np


class Perceptron_2D():
    def __init__(self,num_data=1,learning_rate=0.1):
        self.weight=[]
        self.learning_rate=learning_rate
        self.x=np.zeros((3,num_data))
        self.D=np.zeros((num_data))
        self.k=1
        self.E=0
        self.y=0
        self.number_of_data=num_data
        for i in range(num_data):
            self.x[0][i]=1
        pass
    
    def Initialize_data(self,matrix_2_num):
        for i in range(self.number_of_data):
            self.x[0][i]=1
        for i in range(2):
            for j in range(self.number_of_data):
                self.x[i+1][j]=matrix_2_num[i][j]

    def Initialize_expected_result(self,matrix_1_num):
        for i in range(self.number_of_data):
            self.D[i]=matrix_1_num[i]

    def Initialize_weights(self):
        self.weight=np.random.rand(3)
        self.weight[0]*=-1
    
    def Initialize_training_parameter(self):
        self.E=0
        self.k=1

    def Output(self,weight,x):
        step=(weight.T)@x
        if(step>=0):
            self.y=1
        else:
            self.y=0

    def Weight_Update(self,D,y,x):
        self.weight=self.weight+self.learning_rate*(D-y)*x


    def Loss_update(self,D,y):
        self.E=self.E +1/2*np.power((D-y),2)

    def Learning_process(self):
        self.Initialize_weights()
        self.Initialize_training_parameter()

        while(True):
            self.Output(self.weight,self.x[:,(self.k-1)])
            self.Weight_Update(self.D[self.k-1],self.y,self.x[:,(self.k-1)])

            self.Loss_update(self.D[self.k-1],self.y)

            if self.k<self.number_of_data:
                self.k+=1
                continue
            else:
                pass

            if self.E < 0.05:
                break
            else:
                print(self.weight)
                print(self.E)
                self.Initialize_training_parameter()  

           

        

