from perceptron import *
import numpy as np
import matplotlib.pyplot as plt

perceptron=Perceptron_2D(4,0.5)

data=np.array([[0,0,1,1],[0,1,0,1]])

perceptron.Initialize_data(data)

perceptron.Initialize_expected_result(np.array([0,1,1,1]))

perceptron.Learning_process()

output=[]
for i in range(np.arange(-1,2,0.1).shape[0]):
    output.append(-perceptron.weight[0]/perceptron.weight[2]-perceptron.weight[1]/perceptron.weight[2]*(-1+i*0.1))

output=np.array(output)


plt.figure()
plt.plot(data[0,0],data[1,0],'or')
plt.plot(data[0][1:4],data[1][1:4],'ob')
plt.plot(np.arange(-1,2,0.1),output)
plt.xlim([-1,2])
plt.ylim([-1,2])
plt.grid(True)
plt.show()




