import numpy as np 
from scipy import integrate
import scipy.constants
import cmath
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# all parameters 

t = [] # list to store time 
out_list = [] # list to store auto-correlation values
clc = 0
p = 1
ff = []
cho = []
qho1 = []
qho2 = []
y = [] # store DOS overall
ys = [] # store DOS solid contribution
yg = [] # store DOS gas contribution
m = 16.04*(1e-3)/scipy.constants.Avogadro # mass of one methane molecule
N = 10 # number of particles in the system
vol = 4.00440*3.97980*4.01490*(1e-27) # volume of the system
den = m*N/vol # density of the system
#print(den)
#-----------------------------------------------------------

# reading file

data = np.loadtxt('vac-CH4-308K-10-RUN1-prod1.xvg',skiprows = 17)
#-----------------------------------------------------------

# storing time and correlation values to corresponding lists

length = len(data) # calculating length of the data lists
for i in range(0,length):
	out_list.append(data[i][1]) # storing auto correlation values
	t.append(data[i][0]) # storing time
#-----------------------------------------------------------

"""
# performing discrete fourier cosine transform on auto correlation values (1)

for i in range(length):
    for j in range(length):
        angle = 2*np.pi*float(i*j/length)
        clc = clc + out_list[j]*np.cos(angle)
    y.append(clc)
    clc = 0
print(y)
#-----------------------------------------------------------
"""


# performing discrete fourier cosine transform on auto correlation values (2)

y = dct(out_list,2)
#-----------------------------------------------------------

# calculating total degrees of freedom and self diffusion coefficient of the system

for i in range(len(y)):
    y[i] = (y[i])*4*(5e-15)/(308*scipy.constants.k)
    y[i] = y[i]*m*(1e6)
    ff.append(i/((5e-15)*len(y)))
tdos = integrate.simps(y,ff) # total number of degrees of freedom
Ds= y[0]*(308*scipy.constants.k)/(12*m*N) # Self diffusion coefficient
#print(Ds)
#print(tdos)
#plt.plot(ff,y) # plotting S(v) vs v
#plt.show()
#-----------------------------------------------------------

#calculate solid and gas fraction of dos (here y)


"""
# calcultauing parttition function using classical harmonic oscillator

for i in range(1,len(xf)):
	ff.append((i / (len(xf) * 5)))
for i in range(len(ff)):
	cho.append(y[p] / ff[i])
	qho1.append(y[p] * ff[i])
	qho2.append(y[p])
	p += 1
#print(cho)
#print(ff)
chopf = integrate.simps(cho,ff)
print(chopf)
qhopf1 = - integrate.simps(qho1,ff)
qhopf2 = - integrate.simps(qho2,ff)
print(qhopf1)
print(qhopf2)
#-----------------------------------------------------------
"""
