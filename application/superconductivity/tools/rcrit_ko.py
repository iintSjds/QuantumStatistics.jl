import os
import sys
import numpy as np
import scipy.stats as st
import scipy.optimize as op
from matplotlib import pyplot as plt

plt.style.use(['science','grid'])

def fit(x,a,b,rc): 
    return a*x*(x-rc)/(1+b*x)

mus = [
    [0.0,0.00077,0.00161, 0.00546, 0.0263, 0.0602, 0.129],
    [0.00957,0.0504,0.0950,0.141,0.188,0.239,0.342],
    [0.0356,0.0698,0.105,0.138,0.186,0.228,0.325],
    [0.0489,0.0802,0.111,0.148,0.188,0.224,0.311]
]

ls = [0, 1, 2, 3]
rss = [1.5,2.0,2.5, 3.0, 3.5, 4.0, 5.0]
rss = np.array(rss)

plt.figure()
colors = ["r","g","b","c","m","y","k"]
plt.title(r"$r_s$ vs $-\mu^{*}$")
for i in range(len(ls)):
    pcolor=colors[i%len(colors)]
    plt.plot(rss, mus[i],marker="o",linestyle="",markersize=2, label = r"$l=%d$"%ls[i],color=pcolor )
    cut = 0
    if ls[i]==0:
        cut = 3
    popt, pcov = op.curve_fit(fit, rss[cut:],mus[i][cut:],p0=(1.0,1.0,0.5))#,sigma=codata[:,2])
    a,b,rc=popt[0],popt[1],popt[2]
    print(popt)
    plt.plot(rss[cut:], fit(rss[cut:],a,b,rc),linestyle="-",color=pcolor, label=r"$a=%.3f,b=%.3f,r_c=%.3f$"%(a,b,rc))


plt.legend(fontsize="xx-small")
plt.xlabel(r"$r_s$")
plt.xlim([0.0,5.0])
plt.ylabel(r"$-\mu^{*}$")
plt.savefig("rsVmu.pdf")
