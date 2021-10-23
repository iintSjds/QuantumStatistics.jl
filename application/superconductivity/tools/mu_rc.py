import os
import sys
import numpy as np
import scipy.stats as st
import scipy.optimize as op
from matplotlib import pyplot as plt

plt.style.use(['science','grid'])



def collect_mu(foldname, llist):
    def fit(x,b,k): 
        return k*x+b
    print(foldname)
    dup=1
    size0=dup*16
    size=4*size0
    beta0=400
    beta=400
    r_s=2.0
    channel=-1
    label=0
    mom_sep0 = 0.00005
    mom_sep= 0.0001
    freq_sep = 2.0
    ll = np.zeros(size // size0)
    tcs = np.zeros(size // size0)
    betas = np.zeros(size0)
    lamus = np.zeros((size // size0, size0))
    lamuserr = np.zeros((size // size0, size0))

    if(len(sys.argv)==2):
        label=int(sys.argv[1])
    for k in range(1,size+1):
        beta = beta * np.sqrt(2)
        if(k%size0==1):
            channel=channel+1
            ll[(k-1) // size0] = channel
            beta = beta0
        betas[(k - 1) % size0] = beta
        try:
            fname = "lamu_{0}.dat".format(k)
            fo = open(foldname+"{0}/".format(k)+fname,"r")
            fsp = fo.read().split()
            lamus[(k-1)//size0, (k-1)%size0]=float(fsp[-1])
            lamuserr[(k-1)//size0, (k-1)%size0]=np.abs(float(fsp[-1])-float(fsp[-2]))
        except IOError:
            lamus[(k - 1) // size0, (k - 1) % size0] = 0.0
    lamus = lamus[:, :-2]
    lamuserr = lamuserr[:, :-2]
    lnbetas = np.log10(betas)[:-2]
    errs = np.zeros(size // size0)
    errs2 = np.zeros(size // size0)
    mu = np.zeros(size // size0)
    errmu = np.zeros(size // size0)

    for nl in llist:
        lamul = lamus[nl, :]
        lamulerr = lamuserr[nl, :]
        codata = np.array([[lnbetas[i], lamul[i],lamulerr[i]] for i in range(len(lnbetas)) if lamul[i]!=0.0 ])
        half = len(codata)//2
        popt, pcov = op.curve_fit(fit, codata[:,0],codata[:,1],p0=(0.0,0.0),sigma=codata[:,2])
        k, b = popt[1], popt[0]
        popt2, pcov2 = op.curve_fit(fit, codata[half:,0],codata[half:,1],p0=(0.0,0.0),sigma=codata[half:,2])
        k2, b2 = popt2[1], popt2[0]
        ek, eb = np.sqrt(pcov[1,1]), np.sqrt(pcov[0,0])
        tc = 1.0/10.0**((1.0-b)/k)
        tc2 = 1.0/10.0**((1.0-b2)/k2)
        tcs[nl] = tc
        errs[nl] = np.sqrt((k*eb)**2+(b*ek)**2)/k**2
        errs2[nl] = tc2#np.abs(tc-tc2)
        mu[nl] = k
        errmu[nl] = ek
    return mu, errmu

folds = [
    "./ko__g0w0__rs1_5__flow/",
    "./ko__g0w0__rs2_0__flow/",
    "./ko__g0w0__rs2_5__flow/",
    "./ko__g0w0__rs3_0__flow/",
    "./ko__g0w0__rs3_5__flow/",
    "./ko__g0w0__rs4_0__flow/",
    "./ko__g0w0__rs5_0__flow/",
]
rss = [1.5,2.0,2.5, 3.0, 3.5, 4.0, 5.0]
rss = np.array(rss)

mus = []
errmus = []

for i in range(len(rss)):
    llist=[0,1,2,3]
    if i == 0:
        llist=[1,2,3]
    mu, errmu = collect_mu(folds[i],llist)
    mus.append(mu)
    errmus.append(errmu)

mus = np.array(mus).T
errmus=np.array(errmus).T
print(mus)
print(errmus)
ls = [0, 1, 2, 3]

def fit(x,a,b,rc): 
    return a*x*(x-rc)/(1+b*x)

plt.figure()
colors = ["r","g","b","c","m","y","k"]
plt.title(r"$r_s$ vs $-\mu^{*}$")
for i in range(len(ls)):
    pcolor=colors[i%len(colors)]
    cut = 0
    if ls[i]==0:
        cut = 3
    plt.errorbar(rss[cut:], mus[i][cut:],errmus[i][cut:],marker="o",fillstyle="none",linestyle="",ms=2,mew=1, label = r"$l=%d$"%ls[i],color=pcolor )
    popt, pcov = op.curve_fit(fit, rss[cut:],mus[i][cut:],p0=(1.0,1.0,0.5))#,sigma=errmus[i][cut:])
    a,b,rc=popt[0],popt[1],popt[2]
    print(popt)
    #plt.plot(rss[cut:], fit(rss[cut:],a,b,rc),linestyle="-",color=pcolor, label=r"$a=%.3f,b=%.3f,r_c=%.3f$"%(a,b,rc))
    plt.plot(rss, fit(rss,a,b,rc),linestyle="-",color=pcolor, label=r"$a=%.3f,b=%.3f,r_c=%.3f$"%(a,b,rc))

plt.legend(fontsize="xx-small")
plt.xlabel(r"$r_s$")
plt.xlim([0.0,5.0])
plt.ylabel(r"$-\mu^{*}$")
plt.savefig("rsVmu.pdf")
