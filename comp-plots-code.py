
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

infile = sys.argv[1] # system argument to specify input file
timefile = sys.argv[2]
reffile = sys.argv[3] # system argument to specify reference file
refpropfile = sys.argv[4] # system argument to specify reference state properties file

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

# all parameters and constants

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
yghs = []
ygwi = []
yswi = []
yshs = []
#--------------------------------------------------------------------------------------------------------------------

# storing time and correlation values to corresponding lists

N = int(numstr) # number of components in the system
den = N/(vol) # particle number density of the system
length = len(data) # calculating length of the data lists
out_list = data[0:]*m*N*(1e6) # storing mass weighted auto correlation values
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

yn = yn*(beta)*2*dt # multiplying by the factor (2/kT)*dt
y = y*(beta)*2*dt # multiplying by the factor (2/kT)*dt
ff = np.arange(length, dtype = np.float_)/(length*dt) # v (frequency)
tdos = integrate.simps(y[:(int)((len(y))/2)],ff[:(int)((len(y))/2)]) # total number of degrees of freedom
dif_sp = y[0]*(T*kb)/(12*m*N) # Self diffusion coefficient (p)

#--------------------------------------------------------------------------------------------------------------------

# Calculating the parameters for the reference state

yref = np.loadtxt(fname = reffile)
dataref = np.loadtxt(fname = refpropfile) 
enermdref = dataref[0] # value of reference gas md energy
dif_sp_ref = dataref[1] # value of self-diffusion coefficient for reference state
u = dataref[2] # value of reference gas gibbs free energy
ent_ref = dataref[3] # value of reference gas entropy

# Calculating separate total solid and gas dos

fl = dif_sp/dif_sp_ref # fluidicity
for i in range(len(y)):
    ygwi.append(fl*yref[i]*N) # calculating gas DOS function vs frequency
    yswi.append(y[i]-ygwi[i]) # calculating solid DOS function vs frequency

# Solid and gas dos vs frequency also calculated here

delta = math.sqrt((np.pi*kb*T)/m)*((6/np.pi)**(2/3))*(den**(1/3))*(2*y[0])/(9*N) # normalized diffusivity constant

def fluidicity(f):
    return (2*(delta**(0-9/2))*(f**(15/2))-(delta**(0-3/2))*(f**(7/2))-6*(f**(5))*(delta**(0-3))+2*f+6*(delta**(0-3/2))*(f**(5/2))-2)
f_value = optimize.root_scalar(fluidicity,bracket = [0,1],method = 'brentq')
frac = f_value.root

pf_assval = (frac**(3/2))*(delta**(-3/2)) # packing fraction
dia = (pf_assval*6/(np.pi*den))**(1/3) # hard sphere diameter
dif_hsfp = (kb*T*y[0])/(12*m*N*frac) # diffusivity of the gas component assuming gas as a hard sphere (fp)
dif_cep = (3/8)*math.sqrt((kb*T)/(np.pi*m))/(den*dia**2) # hard sphere Chapman Enskog diffusion (p)
dif_cefp = dif_cep/frac # hard sphere Chapman Enskog diffusion (fp)

# Calculating separate total solid and gas dos

for i in range(len(y)):
    p = (np.pi*y[0]*ff[i])/(2*frac*tdos)
    yghs.append(y[0]/(1+(p**2))) # calculating gas DOS function vs frequency
    yshs.append(y[i]-yghs[i]) # calculating solid DOS function vs frequency

plt.figure(1)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in y],label = "total")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yghs],label = "gas hard sphere")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yshs],label = "solid hard sphere")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in ygwi],label = "gas widom insertion")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yswi],label = "solid widom insertion")
plt.xlim(0,175)
plt.legend()
#plt.show()
fig = plt.figure(1)
fig.savefig(numstr + "-" + numrun + "-DOS-comp-plot.png")

