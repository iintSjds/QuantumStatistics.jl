import numpy as np
from matplotlib import pyplot as plt
import os
import sys

plt.style.use(['science','grid'])

data_dir="."

filerange = range(96)

if len(sys.argv)==2:
    data_dir=sys.argv[1]

files = [data_dir+"/%d/e_scale.dat"%(i+1) for i in filerange  ]

print(files)

data = []


for fname in files:
    try:
        f = open(fname).read()
        if len(f.split())==3:
            data.append( [float(x) for x in f.split()] )
        print(data[-1])
    except RuntimeError as e:
        print(e)
        pass
    except IOError as e:
        print(e)
        pass

data = np.array(data)
print(data.shape)
rs = list(set(data[:,0]))
print(rs)

la, le = 0, 10
plt.figure()
for r in rs[0:-2]:
    xy = [[data[i,1], data[i,2]] for i in range(data.shape[0]) if data[i,0]==r]
    xy = np.array(xy)
    ia, ie = np.searchsorted(xy[:,0],la),np.searchsorted(xy[:,0],le)
    x = xy[ia:ie,0]
    y = xy[ia:ie,1]
    plt.plot(x,y,"x--", label = "rs=%f"%r)
    print([(2*y[i+1]**(-2)-y[i]**(-2)-y[i+2]**(-2)) for i in range(len(y)-2)])

plt.legend(fontsize="xx-small")
plt.xlabel("$\ell$")
plt.ylabel("$\omega_c / E_F$")
plt.savefig("scale.pdf")

