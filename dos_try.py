"""
Below is the code for computing thermodynamic parameters of a 2PT system
Considered 2 systems - system1 with 10 molecules and system2 with 30 molecules
Both systems at same temperature ie 308K and vol
Can solve for one system by hashing out the other system ( eg solving for system1, hash out every syntax related to system2)
Initially both are hashed out
"""

import glob
import numpy as np 
from scipy import integrate
import scipy.constants
import cmath
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import optimize

c = 0 # counter variable
numstr = []
numrun = []

# reading files
# using glob to open files

filenames = glob.glob('vac-CH4-308K-*-RUN*-prod1.xvg')
numfiles = len(filenames)
for i in range(numfiles):
    dummy = filenames[i]
    lendummy = len(dummy)
    for j in range(lendummy):
        if((dummy[j] == 'K') and (dummy[j+1] == '-')):
            u1 = j + 2
        elif((dummy[j] == 'R') and (dummy[j+1] == 'U')):
            u2 = j - 1
            u3 = j + 3
        elif((dummy[j] == 'p') and (dummy[j+1] == 'r')):
            u4 = j - 1
        else:
            continue
    numstr.append(dummy[u1:u2])
    numrun.append(dummy[u3:u4])

for files in filenames:
    data = np.loadtxt(fname = files,skiprows = 17)

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
    vol = 4.00440*3.97980*4.01490*(1e-27) # volume of the system
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

    N = int(numstr[c]) # number of components in the system
    den = N/vol # particle number density of the system
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

    y = np.empty(length, dtype = np.float_) # store DOS overal
    y = fft(out_list)
    
    #--------------------------------------------------------------------------------------------------------------------

    # calculating total degrees of freedom and self diffusion coefficient of the system 

    y = np.real(y)*(beta)*2*(5e-15) # multiplying by the factor (2/kT)*dt
    ff = np.arange(length, dtype = np.float_)/(length*5e-15) # v (frequency)
    tdos = integrate.simps(y[:(int)((len(y)-1)/2)],ff[:(int)((len(y)-1)/2)]) # total number of degrees of freedom
    dif_sp = y[0]*(T*kb)/(12*m*N) # Self diffusion coefficient (p)
    #plt.plot(ff,y) # plotting S(v) vs v
    #plt.show()

    #--------------------------------------------------------------------------------------------------------------------

    # Calculate solid and gas fraction of dos (here y) along with several diffusion coefficients
    # Solid and gas dos vs frequency also calculated here

    delta = math.sqrt((np.pi*kb*T)/m)*((6/np.pi)**(2/3))*(den**(1/3))*(2*y[0])/(9*N) # normalized diffusivity constant

    # Snippet used to calculate packing fraction and fraction of gas

    def packing_fraction(pf):
        return (2*(delta**3)*(pf**5)-(pf**(7/3))*(delta**2)-6*(pf**(10/3))*(delta**2)+2*delta*(pf**(2/3))+6*delta*(pf**(5/3))-2)
    pf_value = optimize.root_scalar(packing_fraction,x0 = 0,x1 = 1)
    pf_assval = pf_value.root

    # use either of the two values based on the system of interest

    dia = (pf_assval*6/(np.pi*den))**(1/3) # hard sphere diameter
    frac = delta*pf_assval**(2/3) # fraction of gas
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
    tdosg = integrate.simps(yg[:(int)((len(y)-1)/2)],ff[:(int)((len(y)-1)/2)]) # total number of gas DOS
    tdoss = integrate.simps(ys[:(int)((len(y)-1)/2)],ff[:(int)((len(y)-1)/2)]) # total number of solid DOS

    #--------------------------------------------------------------------------------------------------------------------

    # finding partition function for the system and different thermodynamical properties

    # partition function for the solid component

    for i in range(1,len(y)):
        rc.append(0 - ys[i]*np.log(beta*h*ff[i]))
        k = 0 - (beta*h*ff[i])/2 - np.log(1 - np.exp(0 - (beta*h*ff[i]))) # natural logarithm of quantum harmonic oscillator
        rq.append(ys[i]*k)
    lchos = integrate.simps(rc[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]) # calculating natural logarithm of solid DOS partition function assuming classical harmonic oscillator
    lqhos = integrate.simps(rq[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]) # calculating natural logarithm of solid DOS partition function assuming quantum harmonic oscillator
    chos = np.exp(lchos) # finding corresponding partition function
    qhos = np.exp(lqhos) # finding corresponding partition function

    # finding entropy contribution for hard sphere fluid

    # Step 1 : find excess entropy contribution per molecule assuming monatomic ideal gas at same temperature and density (united atom model), per molecule basis 

    r = (((2*np.pi*m*kb*T)/(h**2))**(1.5))*(np.exp(2.5))/den
    si = kb*np.log(r) # entropy value per molecule basis

    # Step 2 : use Carnahan Starling equation to calculate the hard sphere excess entropy, per molecule basis

    c1 = frac*pf_assval
    z = (1 + c1 + c1**2 - c1**3)/(1 - c1)**3 # Carnahan Starling EOS
    r1 = np.log(z)
    r2 = (c1*(3*c1-4))/((1-c1)**2)
    r3 = r1 + r2
    sh = si + kb*r3 # excess entropy value of hard sphere fluid per molecule basis

    # Calculating Helmholtz free energy of the system

    W_Ahs = 0.5 - (sh/(3*kb)) # weighting function for helmholtz free energy corresponding to hard sphere gas
    for i in range(1,(int)((len(y)-1)/2)):
        W_Acho.append(np.log(beta*h*ff[i]))
        ysm1.append(W_Acho[i]*ys[i])
        ygm.append(W_Ahs*yg[i])
        q = (1 - np.exp(0 - beta*h*ff[i]))/np.exp(0 - beta*h*ff[i]/2)
        W_Aqho.append(np.log(q))
        ysm2.append(W_Aqho[i]*ys[i])
    A_cho = (integrate.simps(ygm[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]) + integrate.simps(ysm1[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]))*beta**(-1) # Helmholtz free energy (assuming classical harmonic oscillator)
    A_qho = (integrate.simps(ygm[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]) + integrate.simps(ysm2[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]))*beta**(-1) # Helmholtz free energy (assuming quantum harmonic oscillator)

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
    S_cho = (integrate.simps(ygm[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]) + integrate.simps(ysm1[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]))*kb # Entropy (assuming classical harmonic oscillator)
    S_qho = (integrate.simps(ygm[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]) + integrate.simps(ysm2[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]))*kb # Entropy (assuming quantum harmonic oscillator)

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
    E_cho = (integrate.simps(ygm[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]) + integrate.simps(ys[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]))*(beta**(-1)) # Energy (assuming classical harmonic oscillator)
    E_qho = (integrate.simps(ygm[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]) + integrate.simps(ysm2[1:(int)((len(y)-1)/2)],ff[1:(int)((len(y)-1)/2)]))*(beta**(-1)) # Energy (assuming quantum harmonic oscillator)

    #--------------------------------------------------------------------------------------------------------------------

    # Printing results in a file

    out = open("output" + numstr[c] + "RUN" + numrun[c] + ".txt", "w")
    out.truncate(0)
    print("Data collected for N = " + numstr[c] + " system for RUN = " + numrun[c] + "\n", file = out)
    print("Total Density of states in the system = ", tdos, file = out)
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
    #print("Entropy contribution per molecule assuming monatomic ideal gas at given temperature and density (united atom model), per molecule basis = ", si, file = out)
    #print("Carnahan Starling equation calculated hard sphere entropy, per molecule basis = ", sh, file = out)
    print("Helmholtz free energy of the system (assuming classical harmonic oscillator for solid part of the system) = ", A_cho, file = out)
    print("Helmholtz free energy of the system (assuming quantum harmonic oscillator for solid part of the system) = ", A_qho, file = out)
    print("Entropy of the system  (assuming classical harmonic oscillator for solid part of the system) = ", S_cho, file = out)
    print("Entropy of the system (assuming quantum harmonic oscillator for solid part of the system) = ", S_qho, file = out)
    print("Energy of the system (assuming classical harmonic oscillator for solid part of the system) = ", E_cho, file = out)
    print("Energy of the system (assuming quantum harmonic oscillator for solid part of the system) = ", E_qho, file = out)
    out.close()

    #--------------------------------------------------------------------------------------------------------------------
   
    # Incrementing counter variable for next file read

    c += 1

    #--------------------------------------------------------------------------------------------------------------------



