
# Below is the code for computing thermodynamic parameters of a 2PT system

import sys
import numpy as np 
from scipy import integrate
from scipy import signal
import scipy.constants
import cmath
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import optimize
from scipy.optimize import curve_fit

infile = sys.argv[1] # system argument to specify input file
timefile = sys.argv[2]
enerfile = sys.argv[3] # system argument to specify energy of md simulations
outfile1 = sys.argv[4] # system argument to specify ref dos file
outfile2 = sys.argv[5] # system argument to specify reference state properties file

# Snippet to get run number and number of particles in the system

numstr = 0 # string to store number of components in the systems tested
numrun = 0 # string to store run number for the system being tested

for j in range(len(infile)):
    if((infile[j] == 'K') and (infile[j+1] == '-')):
        u1 = j + 2
    elif((infile[j] == 'a') and (infile[j+1] == 'v')):
        u2 = j - 1
        u3 = j + 3
    elif((infile[j] == 'p') and (infile[j+1] == 'r')):
        u4 = j - 1
    else:
        continue
numstr = infile[u1:u2]
numrun = infile[u3:u4]

#--------------------------------------------------------------------------------------------------------------------

# reading file and performing operations

data = np.loadtxt(fname = infile)
t = np.loadtxt(fname = timefile)

#enermd = np.loadtxt(fname = enerfile,skiprows = 6,usecols = 2) # value of md energy

# all parameters and constants

#t = [] # list to store time 
out_list = [] # list to store auto-correlation values
kb = scipy.constants.k # Boltzmann Constant k
Na = scipy.constants.Avogadro # Avogadro's number
T = 308 # system temperature
ff = [] # storing frequency values
rc = [] # integral for partition function corresponding to classical harmonic oscillator
rq = [] # integral for partition function corresponding to quantum harmonic oscillator
y = [] # store DOS overall
m = 16.04*(1e-3)/Na # mass of one methane molecule
vol = 4.00440*3.97980*4.01490*(1e-27) # volume of the system
beta = 1/(kb*T)
h = scipy.constants.Planck # Planck's constant
enermd = float(enerfile) 
A_ig = []

#--------------------------------------------------------------------------------------------------------------------

# storing time and correlation values to corresponding lists

N = int(numstr) # number of components in the system
den = N/(vol) # particle number density of the system
length = len(data) # calculating length of the data lists
out_list = data[0:]*m*N*(1e6) # storing mass weighted auto correlation values
#t = time # storing time
dt = (t[1]-t[0])*(1e-12) # time step length

#--------------------------------------------------------------------------------------------------------------------

# Normalizing data such that C(0) = 3*N*kb*T

out_list = (out_list/out_list[0])*3*N*kb*T
out_list = np.hstack((out_list,out_list[::-1])) # making autocorrelation values symmetric
length *= 2 

#-------------------------------------------------------------------------------------------------------------------- 
"""
# performing discrete fourier cosine transform on auto correlation values (1)(takes a lot of time to run and is similar to dct in python)

angles = 2*np.pi*np.arange(length, dtype=np.float_)/length
y = np.empty(length, dtype = np.float_) # store DOS overal
for i in range(length):
    y[i] = np.sum(out_list*np.cos(i*angles)) 
#print(y)

#--------------------------------------------------------------------------------------------------------------------
"""

# performing discrete fourier transform on auto correlation values using inbult python functions(2)

yn = np.empty(length, dtype = np.float_) # store DOS overal
yn = fft(out_list) 
yn = np.real(yn)
y = signal.savgol_filter(yn,21,5)

#--------------------------------------------------------------------------------------------------------------------

# calculating total degrees of freedom and self diffusion coefficient of the system 

yn = yn*(beta)*2*dt/60 # multiplying by the factor (2/kT)*dt
y = y*(beta)*2*dt/60 # multiplying by the factor (2/kT)*dt
ff = np.arange(length, dtype = np.float_)/(length*dt) # v (frequency)
tdos = integrate.simps(y[:(int)((len(y))/2)],ff[:(int)((len(y))/2)]) # total number of degrees of freedom
dif_sp = y[0]*(T*kb)/(12*m*1) # Self diffusion coefficient (p)
print(tdos)

out = open(outfile1,"w")
out.truncate(0)
for i in range(length):
    print(y[i], file = out)
out.close()

# finding partition function for the system and different thermodynamical properties

# calculating reference energy

enermd = float(enermd)*(1e3)/Na + 1.5*kb*T # converting to appropriate units
u_ex = -8.66171*(1e3)/Na # excess free energy
r = (((2*np.pi*m*kb*T)/(h**2))**(1.5))*(vol/1)
u_ig = -kb*T*np.log(r) # ideal free energy
u = u_ig + u_ex # total free energy
s = -(u - enermd)/T

Nr = np.linspace(1,180,180)
for i in Nr:
    x = - kb*T*i*np.log(vol*((2*np.pi*m*kb*T)/(h**2))**(1.5))
    for j in range(int(i)):
        x = x + kb*T*np.log(j+1)
    A_ig.append(x)
    x = 0

def func(x, a, b, c, d, e):
    return a + b*x + c*(x**2) + d*(x**3) + e*(x**4) 

popt , pconv = curve_fit(func,Nr,A_ig)
a = popt[0]
b = popt[1]
c = popt[2]
d = popt[3]
e = popt[4]

#print(pconv)
# Printing results in a file
plt.figure(1)
plt.plot(Nr,A_ig,label = "data")
plt.plot(Nr,func(Nr,*popt), label = "curve_fit")
plt.legend()
plt.show()

out = open(outfile2,"w")
out.truncate(0)
print(enermd, file = out)
print(dif_sp, file = out)
print(u_ex, file = out)
print(s, file = out)
print(a, file = out)
print(b, file = out)
print(c, file = out)
print(d, file = out)
print(e, file = out)
out.close()

#--------------------------------------------------------------------------------------------------------------------
print("COMPLETED!!")
