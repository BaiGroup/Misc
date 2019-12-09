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
outfile = sys.argv[2] # system argument to specify output file
enerfile = sys.argv[3] # system argument to specify energy of md simulations

# Snippet to get run number and number of particles in the system

numstr = 0 # string to store number of components in the systems tested
for j in range(len(infile)):
    if((infile[j] == '-')):
        u1 = j + 1
    elif((infile[j] == '.')):
        u2 = j
    else:
        continue
numstr = infile[u1:u2]

#--------------------------------------------------------------------------------------------------------------------

# reading file and performing operations

data = np.loadtxt(fname = infile,skiprows = 17)
enermdkd = np.loadtxt(fname = enerfile,skiprows = 24) # value of md energy
enermdk = enermdkd[:,1]
mde = 0
for i in range(len(enermdk)):
    mde += enermdk[i]
enermd = mde/len(enermdk)

# all parameters and constants

t = [] # list to store time 
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
vol = 4.00440*3.97980*4.01490*(1e-27)*0.104916 # volume of the system
beta = 1/(kb*T)
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
out_list = data[:,1]*m*N*(1e6) # storing mass weighted auto correlation values
t = data[:,0] # storing time
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
y = signal.savgol_filter(yn,401,5)

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

# Calculate solid and gas fraction of dos (here y) along with several diffusion coefficients
# Solid and gas dos vs frequency also calculated here

delta = math.sqrt((np.pi*kb*T)/m)*((6/np.pi)**(2/3))*(den**(1/3))*(2*y[0])/(9*N) # normalized diffusivity constant

# Snippet used to calculate packing fraction and fraction of gas

"""
def packing_fraction(pf):
    return (2*(delta**3)*(pf**5)-(delta**2)*(pf**(7/3))-6*(pf**(10/3))*(delta**2)+2*delta*(pf**(2/3))+6*delta*(pf**(5/3))-2)
pf_value = optimize.root_scalar(packing_fraction,x0 = 0,x1 = 1)
pf_assval = pf_value.root
"""

# Snippet used to calculate packing fraction and fraction of gas

q = 0
def fluidicity(f):
    return (2*(delta**(0-9/2))*(f**(4.5*q+15/2))-6*(delta**(0-3))*(f**(4*q+5))-(delta**(0-3/2))*(f**(2.5*q+7/2))+6*(delta**(0-3/2))*(f**(1.5*q+5/2))+2*(f**(q+1))-2)
f_value = optimize.root_scalar(fluidicity,bracket = [0,1],method = 'brentq')
frac = f_value.root

#frac = delta*(pf_assval**(2/3)) # fraction of gas
pf_assval = (frac**(3/2))*(delta**(-3/2)) # packing fraction
dia = (pf_assval*6/(np.pi*den))**(1/3) # hard sphere diameter
dif_hsfp = (kb*T*y[0])/(12*m*N*frac) # diffusivity of the gas component assuming gas as a hard sphere (fp)
dif_cep = (3/8)*math.sqrt((kb*T)/(np.pi*m))/(den*dia**2) # hard sphere Chapman Enskog diffusion (p)
dif_cefp = dif_cep/frac # hard sphere Chapman Enskog diffusion (fp)

"""
# some verification

c1 = frac*pf_assval
z = (1 + c1 + c1**2 - c1**3)/(1 - c1)**3 # Carnahan Starling EOS
coeff = (4*c1)/(z-1)
k = coeff * dif_cefp/dif_hsfp
print(k)
"""

# Calculating separate total solid and gas dos

for i in range(len(y)):
    p = (np.pi*y[0]*ff[i])/(2*frac*tdos)
    yg.append(y[0]/(1+(p**2))) # calculating gas DOS function vs frequency
    ys.append(y[i]-yg[i]) # calculating solid DOS function vs frequency

flag = 1
for i in range((int)((len(y))/2))[::-1]:
    if ((y[i]-yg[i])>0 and (y[i-1]-yg[i-1]>0)):
        ka = i
        break
    if (y[i]>0 and y[i-1]>0 and flag == 1):
        kc = i    
        flag = 0 
print(ff[int(len(y)/2)]/3e10)
print(ff[ka]/3e10)
print(ff[kc]/3e10)

# pseudo block

tdosg = integrate.simps(yg[:(int)((len(y))/2)],ff[:(int)((len(y))/2)]) # total number of gas DOS
tdoss = integrate.simps(ys[:ka],ff[:ka]) # total number of solid DOS
tdosa = tdoss + tdosg

"""
plt.figure(1)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yg],label = "gas")
plt.plot([p/3e10 for p in ff],[p*3e10 for p in ys],label = "solid")
plt.legend()

plt.figure(1)
plt.plot([p/3e10 for p in ff],[p*3e10 for p in yd],label = "correction")
plt.legend()
"""
#--------------------------------------------------------------------------------------------------------------------

# finding partition function for the system and different thermodynamical properties

# calculating reference energy

enermd = float(enermd)*(1e3)/Na # converting to appropriate units
refener = enermd - ((3*N)*(1 - (0.5*frac))/(beta)) # reference energy

# partition function for the solid component

for i in range(1,len(y)):
    rc.append(0 - ys[i]*np.log(beta*h*ff[i]))
    k = 0 - (beta*h*ff[i])/2 - np.log(1 - np.exp(0 - (beta*h*ff[i]))) # natural logarithm of quantum harmonic oscillator
    rq.append(ys[i]*k)
lchos = integrate.simps(rc[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]) # calculating natural logarithm of solid DOS partition function assuming classical harmonic oscillator
lqhos = integrate.simps(rq[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]) # calculating natural logarithm of solid DOS partition function assuming quantum harmonic oscillator
chos = np.exp(lchos) # finding corresponding partition function
qhos = np.exp(lqhos) # finding corresponding partition function

# finding entropy contribution for hard sphere fluid

# Step 1 : find excess entropy contribution per molecule assuming monatomic ideal gas at same temperature and density (united atom model), per molecule basis 

r = (((2*np.pi*m*kb*T)/(h**2))**(1.5))*(np.exp(2.5))/(den*frac)
si = kb*np.log(r)  # entropy value per molecule basis

# Step 2 : use Carnahan Starling equation to calculate the hard sphere excess entropy, per molecule basis

c1 = frac*pf_assval
z = (1 + c1 + c1**2 - c1**3)/(1 - c1)**3 # Carnahan Starling EOS
r1 = np.log(z)
r2 = (c1*(3*c1-4))/((1-c1)**2)
r3 = r1 + r2
sh = si + kb*r3 # excess entropy value of hard sphere fluid per molecule basis

# Calculating Helmholtz free energy of the system and thus partition function of the system

W_Ahs = 0.5 - (sh/(3*kb)) # weighting function for helmholtz free energy corresponding to hard sphere gas
for i in range(1,(int)((len(y)))):
    W_Acho.append(np.log(beta*h*ff[i]))
    ysm1.append(W_Acho[i]*ys[i])
    ygm.append(W_Ahs*yg[i])
    q = (1 - np.exp(0 - beta*h*ff[i]))/np.exp(0 - beta*h*ff[i]/2)
    W_Aqho.append(np.log(q))
    ysm2.append(W_Aqho[i]*ys[i])
A_cho = refener + (integrate.simps(ygm[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]) + integrate.simps(ysm1[1:ka],ff[1:ka]))*beta**(-1) # Helmholtz free energy (assuming classical harmonic oscillator)
A_qho = refener + (integrate.simps(ygm[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]) + integrate.simps(ysm2[1:ka],ff[1:ka]))*beta**(-1) # Helmholtz free energy (assuming quantum harmonic oscillator)
Q_cho = 0 - (A_cho)/(kb*T) # partition function of the system (assuming classical harmonic oscillator)
Q_qho = 0 - (A_qho)/(kb*T) # partition function of the system (assuming quantum harmonic oscillator)

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
# reinitializing different parameters

ysm1 = [] 
ysm1.append(0)
ysm2 = [] 
ysm2.append(0)
ygm = []
ygm.append(0)

# Calculating Entropy of the system

W_Shs = (sh/(3*kb)) # weighting function for entropy corresponding to hard sphere gas
for i in range(1,len(y)):
    ygm.append(W_Shs*yg[i])
    W_Scho.append(1 - np.log(beta*h*ff[i]))
    ysm1.append(W_Scho[i]*ys[i])
    a1 = (beta*h*ff[i])/(np.exp(beta*h*ff[i])-1)
    a2 = np.log(1 - np.exp(0 - beta*h*ff[i]))
    W_Sqho.append(a1-a2)
    ysm2.append(W_Sqho[i]*ys[i])
S_cho1 = (integrate.simps(ysm1[1:ka],ff[1:ka]))*kb # Entropy (assuming classical harmonic oscillator)
S_qho1 = (integrate.simps(ysm2[1:ka],ff[1:ka]))*kb # Entropy (assuming quantum harmonic oscillator)
S_hs = (integrate.simps(ygm[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]))*kb # Entropy (assuming quantum harmonic oscillator)
S_cho = S_cho1 + S_hs
S_qho = S_qho1 + S_hs

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

ysm1 = [] 
ysm1.append(0)
ysm2 = [] 
ysm2.append(0)
ygm = []
ygm.append(0)

# Calculating Energy of the system

W_Ehs = 0.5 # weighting function for energy corresponding to hard sphere gas
W_Echo = 1 # weighting function for energy corresponding to solid assuming classical harmonic oscillator
for i in range(1,len(y)):
    ygm.append(W_Ehs*yg[i])
    b1 = beta*h*ff[i]/2
    b2 = (beta*h*ff[i])/(np.exp(beta*h*ff[i])-1)
    W_Eqho.append(b1+b2)
    ysm2.append(W_Eqho[i]*ys[i])
E_cho = refener + (integrate.simps(ygm[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]) + integrate.simps(ys[1:ka],ff[1:ka]))*(beta**(-1)) # Energy (assuming classical harmonic oscillator)
E_qho = refener + (integrate.simps(ygm[1:(int)((len(y))/2)],ff[1:(int)((len(y))/2)]) + integrate.simps(ysm2[1:ka],ff[1:ka]))*(beta**(-1)) # Energy (assuming quantum harmonic oscillator)

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
#--------------------------------------------------------------------------------------------------------------------

# Printing results in a file

out = open(outfile,"w")
out.truncate(0)
print("Data collected for N = " + numstr, file = out)
print("All values are in their SI units \n", file = out)
print("Total Density of states in the system = ", tdosa, file = out)
print("Self diffusion coefficient at given density = ", dif_sp, file = out)
print("Normalized diffusivity coefficient / delta = ", delta, file = out)
print("Packing fraction = ", pf_assval, file = out)
print("Hard sphere diameter = ", dia, file = out)
print("Fluidicity / Fraction of gas = ", frac, file = out)
print("Diffusivity of the gas component assuming gas as a hard sphere at frac.density = ", dif_hsfp, file = out)
print("Hard sphere Chapman Enskog diffusion at density = ", dif_cep, file = out)
print("Hard sphere Chapman Enskog diffusion at frac.density = ", dif_cefp, file = out)
print("Total number of gas DOS in the system = ", tdosg, file = out)
print("Total number of solid DOS in the system = ", tdoss, file = out)
print("Solid DOS partition function assuming classical harmonic oscillator = ", chos, file = out)
print("Solid DOS partition function assuming quantum harmonic oscillator = ", qhos, file = out)
print("MD energy of the system scales with N = ", enermd, file = out)
print("Reference energy of the system scales with N = ", refener, file = out)
print("Entropy contribution per molecule assuming monatomic ideal gas at given temperature and density (united atom model), per molecule basis = ", si, file = out)
print("Carnahan Starling equation calculated hard sphere entropy, per molecule basis = ", sh, file = out)
print("Helmholtz free energy of the system (assuming classical harmonic oscillator for solid part of the system) scales with N = ", A_cho, file = out)
print("Helmholtz free energy of the system (assuming quantum harmonic oscillator for solid part of the system) scales with N = ", A_qho, file = out)
print("Entropy of the system (assuming classical harmonic oscillator for solid part of the system) per molecule basis = ", S_cho/(N*kb), file = out)
print("Entropy of the system (assuming quantum harmonic oscillator for solid part of the system) per molecule basis = ", S_qho/(N*kb), file = out)
print("Energy of the system (assuming classical harmonic oscillator for solid part of the system) scales with N = ", E_cho, file = out)
print("Energy of the system (assuming quantum harmonic oscillator for solid part of the system) scales with N = ", E_qho, file = out)
print("Partition function of the system (assuming classical harmonic oscillator for solid part of the system) scales with N = ", Q_cho, file = out)
print("Partition function of the system (assuming quantum harmonic oscillator for solid part of the system) scales with N = ", Q_qho, file = out)
out.close()

#--------------------------------------------------------------------------------------------------------------------

print("System with N = " + numstr + " completed!!") # indication that the particular run is completed
zero = []
j = []
c = 0
for i in range(250):
    zero.append(0)
    j.append(c)
    c += 1
"""
plt.figure(1)
#plt.plot(j,zero)
#plt.xticks(np.arange(0,180,30))
#plt.yticks(np.arange(0,35,5))
plt.show()
"""
