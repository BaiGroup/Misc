import numpy as np
import sys
import scipy.constants
import matplotlib.pyplot as plt

Na = scipy.constants.Avogadro # Avogadro's number
kb = scipy.constants.k # Boltzmann Constant k
T = 308 # system temperature
data1 = sys.argv[1]
data2 = sys.argv[2]
data4 = sys.argv[3]
data3 = []
data1a = np.loadtxt(fname = data1,skiprows = 6,usecols = 2)
data2a = np.loadtxt(fname = data2,skiprows = 6,usecols = 2)
print(data1a)
data3.append((data1a + data2a)*1e3/(2*Na))
out = open(data4,"a")
out.truncate()
for i in range(len(data3)):
	print(data3[i], file = out)
	#print("\n", file = out)
plt.figure(1)
plt.plot()
