import numpy as np 

data1 = np.loadtxt(fname = "Helmholtz-cho_RUN1.txt")
data2 = np.loadtxt(fname = "Helmholtz-cho_RUN2.txt")
Ten_avg_cho = (data1 + data2)/2
out1 = open("Average-Helmholtz-cho.txt","w")
out1.truncate(0)
for i in range(len(data1)):
	print(Ten_avg_cho[i],file = out1)
out1.close()
