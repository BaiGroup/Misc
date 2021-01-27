import numpy as np 
import matplotlib.pyplot as plt
import scipy.constants
import sys

exp_data = sys.argv[1]
kb = scipy.constants.k # Boltzmann Constant k
T = 308 # system temperature
vol = 4.00440*3.97980*4.01490*(1e-27) # volume of the system
Na = scipy.constants.Avogadro # Avogadro's number
h = scipy.constants.Planck # Planck's constant
m = 16.04*(1e-3)/Na # mass of one methane molecule
u_ex = -8.66171*(1e3)/Na # excess gibbs free energy
r = (((2*np.pi*m*kb*T)/(h**2))**(1.5))*kb*T
u0 = 0 - kb*T*np.log(r)
N = []
u_exp = []
u_hen = []
u_diff = []
#P = np.linspace(1e4,1e7,3000)
exp_pr = np.loadtxt(fname = exp_data,skiprows = 1,usecols = 0)
exp_N = np.loadtxt(fname = exp_data,skiprows = 1,usecols = 1)
exp_N = exp_N[0:]*12
c = 0
for i in exp_N:
	u_exp.append(u0 + kb*T*np.log(exp_pr[c]))
	k = i*np.exp(u_ex/(kb*T))*kb*T/vol
	u_hen.append(u0 + kb*T*np.log(k))
	u_diff.append(u_exp[c]-u_hen[c])
	c = c+1 
"""
plt.figure(1)
plt.plot(P,N,label = "Henry's Law approximation")
plt.plot(exp_pr,exp_N,label = "mfi data")
plt.xlabel("Pressure in Pa")
plt.ylabel("Loading")
ax = plt.gca()
ax.set_xscale("log")
plt.legend()
plt.show()
"""
plt.figure(1)
plt.plot(exp_N,u_exp,label = "data given")
plt.plot(exp_N,u_hen,label = "Henry's Law calc.")
#plt.plot(exp_N,u_diff,label = "Difference")
plt.xlabel("Loading")
plt.ylabel("Chemical Potential")
plt.legend()
plt.show()
