import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.interpolate import InterpolatedUnivariateSpline

GCN_cho = []
GCN_qho = []

kb = scipy.constants.k # Boltzmann Constant k
Na = scipy.constants.Avogadro # Avogadro's number
T = 308 # system temperature
beta = 1/(kb*T)
h = scipy.constants.Planck # Planck's constant
m = 16.04*(1e-3)/Na # mass of one methane molecule
N_org = np.arange(10,190,10)
Npre_cho = []
chepot_cho = []
pressure_cho = []
Npre_qho = []
chepot_qho = []
pressure_qho = []
u1 = []
u2 = []
p1 = []
p2 = []
p = []
a1 = []
a2 = []
min1 = []
min2 = []
slope = []
slope.append(0)
sl1 = []
sl1.append(0)
N = np.linspace(0,180,181)
P_cho = []
P_qho = []
Gc = []
Gq = []
Ndc = []
Ndq = []
q1 = []

# chemical potential
r = (((2*np.pi*m*kb*T)/(h**2))**(1.5))*kb*T
u0 = 0 - kb*T*np.log(r)
u11 = (u0 + kb*T*np.log(1e3))
u21 = (u0 + kb*T*np.log(1e7))
u = np.linspace(u11,u21,1000)
out1 = open("chem_pot._values_1e3_1e6_Pa.txt","w")
out1.truncate(0)
for i in range(len(u)):
	print(u[i],file = out1)
out1.close()
#------------------------------------------------------------

datao1 = np.loadtxt("Average_ln_Partition_Function_cho.txt")
#datao2 = np.loadtxt("Average_ln_Partition_Function_qho.txt")
data3 = np.loadtxt("chem_pot._values_1e3_1e6_Pa.txt")
length3 = len(data3)

spl1 = InterpolatedUnivariateSpline(N_org,datao1)
#spl2 = InterpolatedUnivariateSpline(N_org,datao2)
data1 = spl1(N)
#data2 = spl2(N)
length1 = len(data1)
plt.figure(3)
plt.plot(N,data1,label = "lnQ vs N interpolated")
plt.plot(N_org,datao1, label = "lnQ vs N org")
plt.xlabel("N")
plt.ylabel("lnQ")
plt.legend()

for i in range(len(datao1)):
	q1.append(datao1[i]/N_org[i])
splq1 = InterpolatedUnivariateSpline(N_org,q1)
q2 = splq1(N)

plt.figure(7)
plt.plot(N_org,q1,label = "lnQ/N vs N org")
plt.plot(N,q2,label = "lnQ/N vs N interpolated")
plt.legend()

for i in data3:
	r1 = 0	
	r2 = 0
	for j in range(length1):
		r1 += np.exp(data1[j] + beta*i*N[j])
		#r2 += np.exp(data2[j] + beta*i*N[j])
	Gc.append(r1)
	#Gq.append(r2)
out1 = open("GCN_partition_function_cho.txt","w")
out2 = open("ln_GCN_partition_function_cho.txt","w")
out1.truncate(0)
out2.truncate(0)
for i in range(length3):
	print(Gc[i],file = out1)
	print(np.log(Gc[i]),file = out2)
out1.close()
out2.close()
"""
out1 = open("GCN_partition_function_qho.txt","w")
out2 = open("ln_GCN_partition_function_qho.txt","w")
out1.truncate(0)
out2.truncate(0)
for i in range(length3):
	print(Gq[i],file = out1)
	print(np.log(Gq[i]),file = out2)
out1.close()
out2.close()
"""
datau1 = np.loadtxt("GCN_partition_function_cho.txt")
#datau2 = np.loadtxt("GCN_partition_function_qho.txt")
for i in range(length3):
	r1 = 0
	r = []
	#r2 = 0
	for j in range(length1):
		r1 = r1 + (np.exp(data1[j]+beta*data3[i]*N[j]))*N[j]/datau1[i]
		#r2 = r2 + (np.exp(data2[j]+beta*data3[i]*N[j]))*N[j]/datau2[i]
		r.append((np.exp(data1[j]+beta*data3[i]*N[j]))*N[j]/datau1[i])
	Ndc.append(r1)
	#Ndq.append(r2)
p = np.exp((data3[0:]-u0)/(kb*T))
plt.figure(1)
plt.plot(p,Ndc,label = "dis_cho_up")
#plt.plot(p,Ndq,label = "dis_qho")
 
dataAo = np.loadtxt("Average_Helmholtz_cho.txt") 
splA = InterpolatedUnivariateSpline(N_org,dataAo)
dataA = splA(N)

p = [1e4,1e5,1e6,1e7]
u = u0 + kb*T*np.log(p[0:])
for i in range(len(p)):
	r = []
	for j in range(len(dataA)):
		r.append(u[i]*N[j])
	plt.figure(2)
	plt.plot(N,r,label = "uN for P = %d"%p[i] + " Pa")
	plt.xlabel("N")
	plt.ylabel("A or uN")
plt.figure(2)
plt.plot(N,dataA,label = "A")
plt.legend()

for i in range(1,length1-1):
	s1 = (data1[i] - data1[i-1])/(N[i] - N[i-1])
	s2 = (data1[i+1] - data1[i])/(N[i+1] - N[i])
	s3 = (s1 + s2)/2
	slope.append(s3)
slope[0] = (data1[1] - data1[0])/(N[1] - N[0])
slope.append((data1[length1-1] - data1[length1-2])/(N[length1-1] - N[length1-2]))
out1 = open("Slope_lnQ_vs_N_cho.txt","w")
#out1 = open("Slope_lnQ_vs_N_qho.txt","w")
out1.truncate(0)
for i in range(length1):
	print(slope[i],file = out1)
out1.close()

for i in range(1,len(datao1)-1):
	s1 = (datao1[i] - datao1[i-1])/(N_org[i] - N_org[i-1])
	s2 = (datao1[i+1] - datao1[i])/(N_org[i+1] - N_org[i])
	s3 = (s1 + s2)/2
	sl1.append(s3)
sl1[0] = (datao1[1] - datao1[0])/(N_org[1] - N_org[0])
sl1.append((datao1[len(datao1)-1] - datao1[len(datao1)-2])/(N_org[len(datao1)-1] - N_org[len(datao1)-2]))
plt.figure(4)
plt.plot(N,slope,label = "lnQ/N from interpolated lnQ vs N data")
plt.plot(N_org,sl1,label = "lnQ/N from original lnQ vs N data")
plt.xlabel("N")
plt.ylabel("lnQ/N")
plt.legend()

data = np.loadtxt("ln_GCN_partition_function_cho.txt")
length = len(data) 
slope = []
slope.append(0)
for i in range(1,length-1):
	s1 = (data[i] - data[i-1])/(data3[i] - data3[i-1])
	s2 = (data[i+1] - data[i])/(data3[i+1] - data3[i])
	s3 = (s1 + s2)/2
	slope.append(s3)
slope[0] = (data[1] - data[0])/(data3[1] - data3[0])
slope.append((data[length-1] - data[length-2])/(data3[length-1] - data3[length-2]))
out1 = open("Slope_lnGCNpf_vs_chempot_cho.txt","w")
out1.truncate(0)
for i in range(length):
	print(slope[i],file = out1)
out1.close()

data4 = np.loadtxt("Slope_lnGCNpf_vs_chempot_cho.txt")
#data5 = np.loadtxt("Slope_lnGCNpf_vs_chempot_qho.txt")
length4 = len(data4)
for i in range(length4):
	a = data4[i]/beta
	Npre_cho.append(a)
	chepot_cho.append(data3[i])
	pressure_cho.append(np.exp((data3[i]-u0)/(kb*T)))
"""
for i in range(length4):
	a = data5[i]/beta
	Npre_qho.append(a)
	chepot_qho.append(data3[i])
	pressure_qho.append(np.exp((data3[i]-u0)/(kb*T)))
"""
plt.figure(1)
plt.plot(pressure_cho,Npre_cho,label = "cho_GCN_up")
#plt.plot(pressure_cho,Npre_qho,label = "qho_GCN")
plt.xlabel("Pressure in SI units")
plt.ylabel("Loading")

data1 = np.loadtxt(fname = "Slope_lnQ_vs_N_cho.txt")
#data2 = np.loadtxt(fname = "Slope_lnQ_vs_N_qho.txt")
length1 = len(data1)
for i in range(length1):
	u1.append(-kb*T*data1[i])
	#u2.append(-kb*T*data2[i])
	p1.append(np.exp((u1[i]-u0)/(kb*T)))
	#p2.append(np.exp((u2[i]-u0)/(kb*T)))
plt.figure(1)
plt.plot(p1,N, label = "cho_CN_up")
#plt.plot(p2,N, label = "qho_CN")

c = 0
data1 = np.loadtxt(fname = "chem_pot._values_1e3_1e6_Pa.txt")
datar2 = np.loadtxt(fname = "Average_MD_Energy.txt")
datar3 = np.loadtxt(fname = "Average_Entropy_cho.txt")
#datar4 = np.loadtxt(fname = "Average_Entropy_qho.txt")

for i in range(len(datar2)):
	datar3[i] = datar3[i]*N_org[i]
	#datar4[i] = datar4[i]*N_org[i]

spl2 = InterpolatedUnivariateSpline(N_org,datar2)
spl3 = InterpolatedUnivariateSpline(N_org,datar3)
#spl4 = InterpolatedUnivariateSpline(N_org,datar4)
#data4 = spl4(N)
data2 = spl2(N)
data3 = spl3(N)
length1 = len(data1)
length3 = len(data2)
for j in range(length1):
	for i in range(length3):
		a1.append(data2[i] - T*data3[i] - data1[j]*N[i])
		#a2.append(data2[i] - T*data4[i] - data1[j]*N[i])
	if(j==((int)((length1)/10)*c)):
		"""
		plt.figure(2)
		plt.plot(N,a1,label = np.exp((data1[j]-u0)/(kb*T)))
		plt.xlabel("N")
		plt.ylabel("A-uN")
		plt.legend(loc = 'upper left')
		plt.figure(3)
		plt.plot(N,a2,label = np.exp((data1[j]-u0)/(kb*T)))
		plt.xlabel("N")
		plt.ylabel("A-uN")
		plt.legend(loc = 'upper left')	
		"""
		c = c + 1
	min1.append(N[a1.index(min(a1))])
	#min2.append(N[a2.index(min(a2))])
	a1 = []
	#a2 = []
plt.legend(loc = 'upper left')
p1 = np.exp((data1[0:]-u0)/(kb*T))
plt.figure(1)
plt.plot(p1,min1, label = "cho_A-uN_up")
#plt.plot(p1,min2, label = "qho_A-uN")
plt.xlabel("P")
plt.ylabel("loading")
plt.legend()
plt.show()


