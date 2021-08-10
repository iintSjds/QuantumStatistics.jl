import numpy as np
from matplotlib import pyplot as plt
import os
import sys

plt.style.use(['science','grid'])

data_dir="."

filerange = range(96)

if len(sys.argv)==2:
    data_dir=sys.argv[1]

files = [data_dir+"/data/flow_%d.dat"%(i+1) for i in filerange  ]

print(files)

data = []
run = 3


for fname in files:
    try:
        f = open(fname).read()
        if len(f.split())==run+5:
            data.append( [float(x) for x in f.split()] )
        print(data[-1])
    except RuntimeError as e:
        print(e)
        traceback.print_exc()
        pass

data = np.array(data)
print(data.shape)
rs = list(set(data[:,1]))
print(rs)

la, le = 4, 15
plt.figure()
plt.xlabel("$\ell$")
#plt.ylabel("$(\delta\lambda/\lambda)(\ell^4/\ln(\ell)) $")
#plt.yscale("log")
plt.ylabel("$(\delta\lambda/\lambda) $")
xy=[]
x=[]
y=[]
for r in rs[0:-1]:
    xy = [[data[i,3], data[i,run+4], data[i,4]] for i in range(data.shape[0]) if data[i,1]==r]
    xy = np.array(xy)
    #print(xy)
    if(len(xy.shape)==1):
        continue
    ia, ie = np.searchsorted(xy[:,0],la),np.searchsorted(xy[:,0],le)
    #print(ia,ie)
    x = np.array(xy[ia:ie,0])
    #y = xy[ia:ie,1]/xy[ia:ie,2]*x**4/np.log(x)
    y = np.array(xy[ia:ie,1]/xy[ia:ie,2])
    plt.plot(x[4:],y[4:],"x", label = "rs=%f"%r)
    #plt.plot(x,x**(-4)*np.log(x)*0.1)

print(x)    
y2 = 0.1*x**(-4)*np.log(x)
y3 = -0.3*x**(-4)*np.log(x)
#plt.plot(x,y2, label = "$0.1l^{-4}\ln(l)$"%r)
#plt.plot(x,y3, label = "$-0.3l^{-4}\ln(l)$"%r)

plt.legend(fontsize="xx-small")
plt.savefig("kl.pdf")



