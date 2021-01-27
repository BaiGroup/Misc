# for ((i=10;i<=180;i+=10)); do python ../new_approach_scaled_12.py vac-CH4-308K-$i-average-prod1.xvg time.txt ../output-$i.txt Average-MD_Energy.txt ../ref_prop/ref_dos_avg.txt ../ref_prop/ref_prop_avg.txt
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
outfile = sys.argv[3] # system argument to specify output file
enerfile = sys.argv[4] # system argument to specify energy of md simulations
reffile = sys.argv[5] # system argument to specify reference file
refpropfile = sys.argv[6] # system argument to specify reference state properties file

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
enermdall = np.loadtxt(fname = enerfile) # value of md energy

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
ys = [] # store DOS solid contribution
yg = [] # store DOS gas contribution
m = 16.04*(1e-3)/Na # mass of one methane molecule
vol = 4.00440*3.97980*4.01490*(1e-27) # volume of the system
beta = 1/(kb*T)
u_ex = -8.66171*(1e3)/Na
h = scipy.constants.Planck # Planck's constant
W_Acho = [] # weighting function for helmholtz free energy corresponding to solid assuming classical harmonic oscillator
W_Acho.append(0)
W_Aqho = [] # weighting function for helmholtz free energy corresponding to solid assuming quantum harmonic oscillator
W_Aqho.append(0)
W_Scho = [] # weighting function for entropy corresponding to solid assuming classical harmonic oscillator
W_Scho.append(0)
W_Sqho = []  # weighting function for entropy corresponding to solid assuming quantum harmonic oscillator
W_Sqho.append(0)
W_Eqho = [] # weighting function for energy corresponding to solid assuming quantum harmonic oscillator
W_Eqho.append(0)
ysm1 = [] # modified solid DOS parameter used to calculate differnt properties assuming classical harmonic oscillator
ysm1.append(0)
ysm2 = [] # modified solid DOS parameter used to calculate differnt properties assuming quantum harmonic oscillator
ysm2.append(0) 
ygm = [] # modified gas DOS parameter used to calculate different properties
ygm.append(0)

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

"""
plt.figure(1)
plt.plot(ff,yn, label = "org") # plotting S(v) vs v
plt.plot(ff,y, label = "smoothened") # plotting smoothened S(v) vs v
plt.legend()
plt.show()
"""
#--------------------------------------------------------------------------------------------------------------------

# Calculating the parameters for the reference state

yref = np.loadtxt(fname = reffile)
dataref = np.loadtxt(fname = refpropfile) 
enermdref = dataref[0] # value of reference gas md energy
dif_sp_ref = dataref[1] # value of self-diffusion coefficient for reference state
u = dataref[2] # value of reference gas gibbs free energy
ent_ref = dataref[3] # value of reference gas entropy
a = dataref[4] # value of curve parameter for Ideal Helmholtz energy
b = dataref[5] # value of curve parameter for Ideal Helmholtz energy
c = dataref[6] # value of curve parameter for Ideal Helmholtz energy
d = dataref[7] # value of curve parameter for Ideal Helmholtz energy
e = dataref[8] # value of curve parameter for Ideal Helmholtz energy

# Calculating separate total solid and gas dos

fl = dif_sp/dif_sp_ref # fluidicity
for i in range(len(y)):
    yg.append(fl*yref[i]*N) # calculating gas DOS function vs frequency
    ys.append(y[i]-yg[i]) # calculating solid DOS function vs frequency
"""
flag = 1
for i in range((int)((len(y))/2))[::-1]:
    if ((y[i]-yg[i])>0 and (y[i-1]-yg[i-1]>0)):
        ka = i
        break
    if (y[i]>0 and y[i-1]>0 and flag == 1):
        kc = i    
        flag = 0 
"""
tdosg = integrate.simps(yg[:(int)((len(y))/2)],ff[:(int)((len(y))/2)]) # total number of gas DOS
tdoss = integrate.simps(ys[:(int)((len(y))/2)],ff[:(int)((len(y))/2)]) # total number of solid DOS
"""
plt.figure(1)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yg],label = "gas")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in ys],label = "solid")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in y],label = "total")
plt.xlim(0,250)
plt.legend()
fig = plt.figure(1)
"""
"""
plt.figure(1)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yd],label = "correction")
plt.legend()
"""
#--------------------------------------------------------------------------------------------------------------------


# finding partition function for the system and different thermodynamical properties

# calculating reference energy and entropy

enermd = enermdall[int(N/10)-1] # converting to appropriate units
print(enermd)
refener = enermd - ((3*N)*(1 - fl))/(beta) - N*fl*enermdref # reference energy

# partition function for the solid component

for i in range(1,len(y)):
    rc.append(0 - ys[i]*np.log(beta*h*ff[i])) # classical harmonic oscillator
    k = 0 - (beta*h*ff[i])/2 - np.log(1 - np.exp(0 - (beta*h*ff[i]))) # natural logarithm of quantum harmonic oscillator
    rq.append(ys[i]*k)
lchos = integrate.simps(rc[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]) # calculating natural logarithm of solid DOS partition function assuming classical harmonic oscillator
lqhos = integrate.simps(rq[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]) # calculating natural logarithm of solid DOS partition function assuming quantum harmonic oscillator
chos = np.exp(lchos) # finding corresponding partition function
qhos = np.exp(lqhos) # finding corresponding partition function

# Calculating Helmholtz free energy of the system and thus partition function of the system

for i in range(1,(int)((len(y)))):
    W_Acho.append(np.log(beta*h*ff[i]))
    ysm1.append(W_Acho[i]*ys[i])
    q = (1 - np.exp(0 - beta*h*ff[i]))/np.exp(0 - beta*h*ff[i]/2)
    W_Aqho.append(np.log(q))
    ysm2.append(W_Aqho[i]*ys[i])
A_gas = N*fl*u_ex + (a + b*(N*fl) + c*((N*fl)**2) + d*((N*fl)**3) + e*((N*fl)**4))
A_cho1 = refener + (integrate.simps(ysm1[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]))*beta**(-1) # Helmholtz free energy (assuming classical harmonic oscillator)
A_qho1 = refener + (integrate.simps(ysm2[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]))*beta**(-1) # Helmholtz free energy (assuming quantum harmonic oscillator)
A_cho = (A_cho1 + A_gas)
A_qho = (A_qho1 + A_gas)
Q = 0 - (A_cho)/(kb*T) # partition function of the system (assuming classical harmonic oscillator) 
Q_qho = 0 - (A_qho)/(kb*T) # partition function of the system (assuming quantum harmonic oscillator) 

ysm1 = [] 
ysm1.append(0)
ysm2 = [] 
ysm2.append(0)

"""
A = enermd - T*S_cho
Q = -(A)/(kb*T) # partition function of the system (assuming classical harmonic oscillator)
"""
"""
plt.figure(2)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in ysm1],label = "H_solid_cho")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yd1],label = "H_solid_cho_correction")
plt.legend()
plt.figure(3)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in ysm2],label = "H_solid_qho")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yd2],label = "H_solid_qho_correction")
#plt.plot([p/3e10 for p in ff],[p*3e10 for p in ygm],label = "H_gas_hs")
plt.legend()
"""
"""
ysm1 = [] 
ysm1.append(0)
ysm2 = [] 
ysm2.append(0)
ygm = []
ygm.append(0)
"""
# Calculating Energy of the system

W_Echo = 1 # weighting function for energy corresponding to solid assuming classical harmonic oscillator
for i in range(1,len(y)):
    b1 = beta*h*ff[i]/2
    b2 = (beta*h*ff[i])/(np.exp(beta*h*ff[i])-1)
    W_Eqho.append(b1+b2)
    ysm2.append(W_Eqho[i]*ys[i])
E_cho1 = refener + (integrate.simps(ys[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]))*(beta**(-1)) #Total energy (solid part) chm
E_qho1 = refener + (integrate.simps(ysm2[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]))*(beta**(-1)) # Total energy (solid part) qhm
E_cho = N*fl*enermdref + E_cho1 # Energy (assuming classical harmonic oscillator)
E_qho = N*fl*enermdref + E_qho1 # Energy (assuming quantum harmonic oscillator)

"""
plt.figure(6)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in ys],label = "E_solid_cho")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yd],label = "E_solid_cho_correction")
plt.legend()
plt.figure(7)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in ysm2],label = "E_solid_qho")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yd2],label = "E_solid_qho_correction")
#plt.plot([p/3e10 for p in ff],[p*3e10 for p in ygm],label = "E_gas_hs")
plt.legend()
"""

# Calculating Entropy of the system

for i in range(1,len(y)):
    W_Scho.append(1 - np.log(beta*h*ff[i]))
    ysm1.append(W_Scho[i]*ys[i])
    a1 = (beta*h*ff[i])/(np.exp(beta*h*ff[i])-1)
    a2 = np.log(1 - np.exp(0 - beta*h*ff[i]))
    W_Sqho.append(a1-a2)
    ysm2.append(W_Sqho[i]*ys[i])
S_gas = (N*fl*enermdref - A_gas)/T
S_cho1 = (integrate.simps(ysm1[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]))*kb # Entropy (assuming classical harmonic oscillator)
S_qho1 = (integrate.simps(ysm2[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]))*kb # Entropy (assuming quantum harmonic oscillator)
S_cho = (S_cho1 + S_gas)
S_qho = (S_qho1 + S_gas)

"""
plt.figure(4)
plt.plot([p/3e10 for p in ff[1:(int)((len(y))/2)]],[p*3e10 for p in ysm1[1:(int)((len(y))/2)]],label = "S_solid_cho")
plt.plot([p/3e10 for p in ff[1:(int)((len(y))/2)]],[p*3e10 for p in yd1[1:(int)((len(y))/2)]],label = "S_correction_solid_cho")
plt.legend()
plt.figure(5)
plt.plot([p/3e10 for p in ff[1:(int)((len(y))/2)]],[p*3e10 for p in ysm2[1:(int)((len(y))/2)]],label = "S_solid_qho")
plt.plot([p/3e10 for p in ff[1:(int)((len(y))/2)]],[p*3e10 for p in yd2[1:(int)((len(y))/2)]],label = "S_correction_solid_qho")
#plt.plot([p/3e10 for p in ff[1:(int)((len(y))/2)]],[p*3e10 for p in ygm[1:(int)((len(y))/2)]],label = "S_gas_hs")
plt.legend()
"""

# reinitializing different parameters
#--------------------------------------------------------------------------------------------------------------------

# Printing results in a file

out = open(outfile,"w")
out.truncate(0)
print("Data collected for N = " + numstr + " system for RUN = " + numrun, file = out)
print("All values are in their SI units \n", file = out)
print("Total Density of states in the system = ", tdos, file = out)
#print("ka = ", ka, file = out)
print("Fluidicity / Fraction of gas = ", fl, file = out)
print("Total number of gas DOS in the system = ", tdosg, file = out)
print("Total number of solid DOS in the system = ", tdoss, file = out)
print("Solid DOS partition function assuming classical harmonic oscillator = ", chos, file = out)
print("Solid DOS partition function assuming quantum harmonic oscillator = ", qhos, file = out)
print("MD energy of the system scales with N = ", enermd, file = out)
print("Reference energy of the system scales with N = ", refener, file = out)
print("Helmholtz free energy of the system (assuming classical harmonic oscillator for solid part of the system) scales with N = ", A_cho, file = out)
print("Solid Helmholtz free energy contribution (assuming classical harmonic oscillator) scales with N = ", A_cho1, file = out)
print("Gas Helmholtz free energy contribution scales with N = ", A_gas, file = out)
#print("Helmholtz free energy of the system (assuming quantum harmonic oscillator for solid part of the system) scales with N = ", A_qho, file = out)
#print("Helmholtz free energy of the system (assuming classical harmonic oscillator for solid part of the system) scales with N = ", A, file = out)
print("Entropy of the system (assuming classical harmonic oscillator for solid part of the system) per molecule basis = ", S_cho/(N*kb), file = out)
print("Solid Entropy contribution (assuming classical harmonic oscillator) per molecule basis = ", S_cho1/(N*kb), file = out)
print("Gas Entropy contribution per molecule basis = ", S_gas/(N*kb), file = out)
#print("Entropy of the system (assuming quantum harmonic oscillator for solid part of the system) per molecule basis = ", S_qho/(N*kb), file = out)
print("Total Energy of the system (assuming classical harmonic oscillator for solid part of the system) scales with N = ", E_cho, file = out)
print("Solid contribution to Total Energy (assuming classical harmonic oscillator for solid part of the system) scales with N = ", E_cho1, file = out)
print("Gas contribution to Total Energy (assuming classical harmonic oscillator for solid part of the system) scales with N = ", N*fl*enermdref, file = out)
#print("Energy of the system (assuming quantum harmonic oscillator for solid part of the system) scales with N = ", E_qho, file = out)
print("ln of Partition function of the system (assuming classical harmonic oscillator for solid part of the system) scales with N = ", Q, file = out)
#print("ln of Partition function of the system (assuming classical harmonic oscillator for solid part of the system) scales with N = ", Q_cho, file = out)
#print("ln of Partition function of the system (assuming quantum harmonic oscillator for solid part of the system) scales with N = ", Q_qho, file = out)
out.close()

#--------------------------------------------------------------------------------------------------------------------

print("System with N = " + numstr + " and run number = " + numrun + " completed!!") # indication that the particular run is completed
#print("ys[0] = "+ str(ys[0]) + " and " + "yref[0] = "+ str(yref[0]) + " and " + "yg[0] = "+ str(yg[0]) + " and " + "y[0] = "+ str(y[0]))
zero = []
j = []
c = 0
for i in range(250):
    zero.append(0)
    j.append(c)
    c += 1
#fig.savefig(numstr + "-" + numrun + "-DOS-plot.png")
"""
plt.figure(1)
#plt.plot(j,zero)
#plt.xticks(np.arange(0,180,30))
#plt.yticks(np.arange(0,35,5))
plt.show()
"""
