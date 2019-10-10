import numpy as np 
from scipy import integrate
import scipy.constants
import cmath
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from sympy import symbols
from sympy.solvers import solve
from sympy.plotting import plot as symplot

# all parameters and constants

t = [] # list to store time 
out_list = [] # list to store auto-correlation values
kb = scipy.constants.k # Boltzmann Constant k
Na = scipy.constants.Avogadro # Avogadro's number
T = 308 # system temperature
clc = 0
p = 1
ff = [] # storing frequency values
rc = [] # integral for partition function corresponding to classical harmonic oscillator
rq = [] # integral for partition function corresponding to quantum harmonic oscillator
y = [] # store DOS overall
ys = [] # store DOS solid contribution
yg = [] # store DOS gas contribution
m = 16.04*(1e-3)/Na # mass of one methane molecule
N = 10 # number of particles in the system
vol = 4.00440*3.97980*4.01490*(1e-27) # volume of the system
den = N/vol # particle number density of the system
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

# reading file
data = np.loadtxt('vac-CH4-308K-10-RUN1-prod1.xvg',skiprows = 17)

#--------------------------------------------------------------------------------------------------------------------

# storing time and correlation values to corresponding lists

length = len(data) # calculating length of the data lists
for i in range(0,length):
    out_list.append(data[i][1]) # storing auto correlation values
    t.append(data[i][0]) # storing time
#--------------------------------------------------------------------------------------------------------------------

"""
# performing discrete fourier cosine transform on auto correlation values (1)

for i in range(length):
    for j in range(length):
        angle = 2*np.pi*float(i*j/length)
        clc = clc + out_list[j]*np.cos(angle)
    y.append(clc)
    clc = 0
print(y)
#--------------------------------------------------------------------------------------------------------------------
"""


# performing discrete fourier cosine transform on auto correlation values (2)

y = dct(out_list,2)

#--------------------------------------------------------------------------------------------------------------------

# calculating total degrees of freedom and self diffusion coefficient of the system

for i in range(len(y)):
    y[i] = (y[i])*4*(5e-15)*(beta)
    y[i] = y[i]*m*(1e6)
    ff.append(i/((5e-15)*len(y)))
tdos = integrate.simps(y,ff) # total number of degrees of freedom
dif_sp = y[0]*(T*kb)/(12*m*N) # Self diffusion coefficient (p)
#print(dif_sp)
#print(tdos)
# plt.plot(ff,y) # plotting S(v) vs v
# plt.show()

#--------------------------------------------------------------------------------------------------------------------

# Calculate solid and gas fraction of dos (here y) along with several diffusion coefficients
# Solid and gas dos vs frequency also calculated here

delta = math.sqrt((np.pi*kb*T)/m)*((6/np.pi)**(2/3))*(den**(1/3))*(2*y[0])/(9*N) # normalized diffusivity constant
#print(delta)

# Snippet used to iterate packing fraction and fraction of gas
"""
f,pf,eq1,eq2 = symbols('f pf eq1 eq2', real = True) # defining symbols => f = fraction of gas , pf = packing fraction
eq1 = 2*(delta**(-4.5))*(f**7.5)-6*(delta**(-3))*(f**5)-(delta**(-27/4))*(f**(21/4))+6*(delta**(-1.5))*(f**2.5)+2*f-2
eq2 = 2*(delta**3)*(pf**5)-(pf**(7/3))*(delta**2)-6*(pf**(10/3))*(delta**2)+2*delta*(pf**(2/3))+6*delta*(pf**(5/3))-2
pf_val = solve(eq2,check = False, quick = True) # solving equation for pf
print(pf_val)
#f_val = solve(eq1,check = False, quick = True) # solving equation for f # takes too long to solve
#print(f_val)
#symplot(eq1)
#symplot(eq2)S
"""

pf_assval = 5.64288740598082 # packing fraction calculated
dia = (pf_assval*6/(np.pi*den))**(1/3) # hard sphere diameter
frac = delta*pf_assval**(2/3) # fraction of gas
#print(frac)
#print(dia)
dif_hsfp = (kb*T*y[0])/(12*m*N*frac) # diffusivity of the gas component assuming gas as a hard sphere (fp)
#print(dif_hsfp)
dif_cep = (3/8)*math.sqrt((kb*T)/(np.pi*m))/(den*dia**2) # hard sphere Chapman Enskog diffusion (p)
#print(dif_cep)
dif_cefp = dif_cep/frac # hard sphere Chapman Enskog diffusion (fp)
#print(dif_cefp)

""" some verification
c1 = frac*pf_assval
z = (1 + c1 + c1**2 - c1**3)/(1 - c1)**3 # Carnahan Starling EOS
coeff = (4*c1)/(z-1)
k = coeff * dif_cefp/dif_hsfp
print(k)
"""

for i in range(len(y)):
    p = (np.pi*y[0]*ff[i])/(2*frac*12)
    yg.append(y[0]/(1+(p**2))) # calculating gas DOS function vs frequency
    ys.append(y[i]-yg[i]) # calculating solid DOS function vs frequency
tdosg = integrate.simps(yg,ff) # total number of gas DOS
tdoss = integrate.simps(ys,ff) # total number of solid DOS
#print(tdosg)
#print(tdoss)

#--------------------------------------------------------------------------------------------------------------------

# finding partition function for the system and different thermodynamical properties

# partition function for the solid component
for i in range(1,len(y)):
    rc.append(0 - ys[i]*np.log(beta*h*ff[i]))
    k = 0 - (beta*h*ff[i])/2 - np.log(1-np.exp(0 - (beta*h*ff[i])/2)) # natural logarithm of quantum harmonic oscillator
    rq.append(ys[i]*k)
lchos = integrate.simps(rc,ff[1:]) # calculating natural logarithm of solid DOS partition function assuming classical harmonic oscillator
lqhos = integrate.simps(rq,ff[1:]) # calculating natural logarithm of solid DOS partition function assuming quantum harmonic oscillator
chos = np.exp(lchos) # finding corresponding partition function
qhos = np.exp(lqhos) # finding corresponding partition function
# print(chos)
# print(qhos)

# finding entropy contribution for hard sphere fluid

# Step 1 : find entropy contribution per molecule assuming monatomic ideal gas at same temperature and density (united atom model), per molecule basis 
r = (((2*np.pi*m*kb*T)/(h**2))**(1.5))*(np.exp(2.5))/den
si = kb*np.log(r) # entropy value per molecule basis
# print(si)

# Step 2 : use Carnahan Starling equation to calculate the hard sphere entropy, per molecule basis 
c1 = frac*pf_assval
z = (1 + c1 + c1**2 - c1**3)/(1 - c1)**3 # Carnahan Starling EOS
r1 = np.log(z)
r2 = (c1*(3*c1-4))/((1-c1)**2)
r3 = r1 + r2
sh = si + kb*r3 # excess entropy value of hard sphere fluid per molecule basis
# print(sh)

# Calculating helmholtz free energy of the system
W_Ahs = 0.5 - (sh/(3*kb)) # weighting function for helmholtz free energy corresponding to hard sphere gas
for i in range(1,len(y)):
    W_Acho.append(np.log(beta*h*ff[i]))
    ysm1.append(W_Acho[i]*ys[i])
    ygm.append(W_Ahs*yg[i])
    q = (1 - np.exp(0 - beta*h*ff[i]))/np.exp(0 - beta*h*ff[i]/2)
    W_Aqho.append(np.log(q))
    ysm2.append(W_Aqho[i]*ys[i])
A_cho = (integrate.simps(ygm[1:],ff[1:]) + integrate.simps(ysm1[1:],ff[1:]))*beta**(-1) # helmholtz free energy (assuming classical harmonic oscillator)
A_qho = (integrate.simps(ygm[1:],ff[1:]) + integrate.simps(ysm2[1:],ff[1:]))*beta**(-1) # helmholtz free energy (assuming quantum harmonic oscillator)
# print(A_cho)
# print(A_qho)

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
S_cho = (integrate.simps(ygm[1:],ff[1:]) + integrate.simps(ysm1[1:],ff[1:]))*kb # entropy (assuming classical harmonic oscillator)
S_qho = (integrate.simps(ygm[1:],ff[1:]) + integrate.simps(ysm2[1:],ff[1:]))*kb # entropy (assuming quantum harmonic oscillator)
# print(S_cho)
# print(S_qho)

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
    ygm.append(W_Shs*yg[i])
    b1 = beta*h*ff[i]/2
    b2 = (beta*h*ff[i])/(np.exp(beta*h*ff[i])-1)
    W_Eqho.append(b1+b2)
    ysm2.append(W_Eqho[i]*ys[i])
E_cho = (integrate.simps(ygm[1:],ff[1:]) + integrate.simps(ys[1:],ff[1:]))*(beta**(-1)) # energy (assuming classical harmonic oscillator)
E_qho = (integrate.simps(ygm[1:],ff[1:]) + integrate.simps(ysm2[1:],ff[1:]))*(beta**(-1)) # energy (assuming quantum harmonic oscillator)
print(E_cho)
print(E_qho)

#--------------------------------------------------------------------------------------------------------------------