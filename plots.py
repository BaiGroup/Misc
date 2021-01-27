
import numpy as np 
from scipy import optimize
import matplotlib.pyplot as plt
import scipy.constants
import math

q = 0
n1 = np.linspace(-5,4,1000)
delta = [10**p for p in n1]
fv = []
for i in delta:
	def fluidicity(f):
		return(2*(i**(0-9/2))*(f**(4.5*q+15/2))-6*(i**(0-3))*(f**(4*q+5))-(i**(0-3/2))*(f**(2.5*q+7/2))+6*(i**(0-3/2))*(f**(1.5*q+5/2))+2*(f**(q+1))-2)
	f_value = optimize.root_scalar(fluidicity,bracket = [0,1],method = 'brentq')
	fv.append(f_value.root)
plt.figure(1)
plt.semilogx(delta,fv)
plt.xlabel("delta")
plt.ylabel("fluidicity")

kb = scipy.constants.k # Boltzmann Constant k
Na = scipy.constants.Avogadro # Avogadro's number
N = 512
sig = (3.405*1e-10)
den = 0.85/sig**3
m = 39.948*(1e-3)/Na # mass of one argon molecule
ep = 238*4.186/Na
p = ((m/ep)**0.5)/sig
d1 = [0.113,0.153,0.190,0.253]
d2 = [0.069,0.129,0.166,0.203]
d3 = [0.069,0.080,0.100,0.140]
d4 = [0.036,0.049,0.062,0.087]
denp = [0.05,0.40,0.70,0.85]
dk = 0
for d in [d1,d2,d3,d4]:
	c = 0
	far = []
	delta1 = []
	for T in [107.8,131.8,167.7,215.6]:
		y = (d[c]/p)*12*m*N/(kb*T)
		delta1.append(math.sqrt((np.pi*kb*T)/m)*((6/np.pi)**(2/3))*(den**(1/3))*(2*y)/(9*N))
		def fluidicity(f):
			return(2*(delta1[c]**(0-9/2))*(f**(15/2))-(delta1[c]**(0-3/2))*(f**(7/2))-6*(f**(5))*(delta1[c]**(0-3))+2*f+6*(delta1[c]**(0-3/2))*(f**(5/2))-2)
		f_value = optimize.root_scalar(fluidicity,bracket = [0,1],method = 'brentq')
		far.append(f_value.root)
		c = c+1
	plt.figure(1)
	plt.semilogx(delta1,far,label = "density = "+ str(denp[dk]))
	dk += 1

dens = [0.1,0.5,0.9]
for j in dens:
	c = 0
	fv = []
	y = []
	sigma = []
	for i in delta:
		def fluidicity(f):
			return(2*(i**(0-9/2))*(f**(4.5*q+15/2))-6*(i**(0-3))*(f**(4*q+5))-(i**(0-3/2))*(f**(2.5*q+7/2))+6*(i**(0-3/2))*(f**(1.5*q+5/2))+2*(f**(q+1))-2)
		f_value = optimize.root_scalar(fluidicity,bracket = [0,1],method = 'brentq')
		fv.append(f_value.root)
		y.append((fv[c]/i)**1.5)
		sigma.append(((6*y[c])/(np.pi*j/sig**3))**(1/3))
		c = c + 1
	plt.figure(2)
	plt.semilogx(delta,sigma,label = "density = "+ str(j))
	plt.xlabel("delta")
	plt.ylabel("sigma")
	plt.legend()
plt.figure(1)
plt.legend()
plt.figure(2)
plt.show()





