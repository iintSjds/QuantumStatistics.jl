import os
import sys
import numpy as np
import scipy.stats as st
import scipy.optimize as op
from matplotlib import pyplot as plt

plt.style.use(['science','grid'])

def fit(x,b,k): 
    return k*x+b

def extract_flow(dirname, nl):
    dup=1
    size0=dup*16
    size=10*size0
    print(size)
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
    for k in range(1,size+1):
        beta = beta * np.sqrt(2)
        #if(k%dup==1):
        #        beta=beta0
        # r_s=r_s+0.5
        #mom_sep = mom_sep-0.12
        #freq_sep = freq_sep /2.0**0.5
        if(k%size0==1):
            #r_s=0.5
            #mom_sep = 2.0
            #freq_sep = 2.0
            channel=channel+1
            ll[(k-1) // size0] = channel
            beta = beta0
        betas[(k - 1) % size0] = beta
        try:
            fname = "lamu_{0}.dat".format(k)
            fo = open(dirname+"{0}/".format(k)+fname,"r")
            fsp = fo.read().split()
            # print(float(fsp[-1]))
            lamus[(k-1)//size0, (k-1)%size0]=float(fsp[-1])
            lamuserr[(k-1)//size0, (k-1)%size0]=np.abs(float(fsp[-1])-float(fsp[-2]))
        except IOError:
            lamus[(k-1)//size0, (k-1)%size0]=0.0
    # print(ll)
    # print(betas)
    # print(lamus)
    lamus = lamus[:, :-2]
    lamuserr = lamuserr[:, :-2]
    lnbetas = np.log10(betas)[:-2]
    errs = np.zeros(size // size0)
    errs2 = np.zeros(size // size0)

    lamul = lamus[nl, :]
    lamulerr = lamuserr[nl, :]
    codata = np.array([[lnbetas[i], lamul[i],lamulerr[i]] for i in range(len(lnbetas)) if lamul[i]!=0.0 ])
    # if nl == 6:
    #     codata = np.array([[lnbetas[i], lamul[i], lamulerr[i]] for i in range(len(lnbetas)) if i not in [15,12,9]])
    half = len(codata)//2
    # k, b, r_value, p_value, std_err = st.linregress(codata[:,0],codata[:,1])
    # k2, b2, r_value2, p_value2, std_err2 = st.linregress(codata[half:,0],codata[half:,1])
    # tc = 1.0/10.0**((1.0-b)/k)
    # tc2 = 1.0/10.0**((1.0-b2)/k2)
    popt, pcov = op.curve_fit(fit, codata[:,0],codata[:,1],p0=(0.0,0.0))#,sigma=codata[:,2])
    k, b = popt[1], popt[0]
    popt2, pcov2 = op.curve_fit(fit, codata[half:,0],codata[half:,1],p0=(0.0,0.0))#,sigma=codata[half:,2])
    k2, b2 = popt2[1], popt2[0]
    ek, eb = np.sqrt(pcov[1,1]), np.sqrt(pcov[0,0])
    tc = 1.0/10.0**((1.0-b)/k)
    tc2 = 1.0/10.0**((1.0-b2)/k2)
    tcs[nl] = tc
    errs[nl] = np.sqrt((k*eb)**2+(b*ek)**2)/k**2
    errs2[nl] = np.abs(tc-tc2)
    # print(tcs)
    # print(errs)
    # print(errs2)
    return codata[:,0],codata[:,1],codata[:,2],k,b,tc,errs2[nl]



rsvar = "2_0"
rs = 2.0
nl = 3

rsvar = "1_0"
rs = 1.0
nl = 5

rsvar = "0_5"
rs = 0.5
nl = 7

if(len(sys.argv)==2):
    rsvar=int(sys.argv[1])
dirrpa = "./rpa__g0w0__rs" + rsvar + "__flow/"
dirko = "./ko__g0w0__rs"+rsvar+"__flow/"


lnbetas_rpa,lamus_rpa,lamuserr_rpa,k_rpa,b_rpa,tc_rpa,tcerr_rpa = extract_flow(dirrpa,nl)
lnbetas_ko,lamus_ko,lamuserr_ko,k_ko,b_ko,tc_ko,tcerr_ko = extract_flow(dirko,nl)


print(lamus_rpa)
print(lamus_ko)

plt.figure()
plt.title(r"$r_s=%2.2f, l=%d$"%(rs,nl))
crpa, cko = "r", "k"

#plt.errorbar(lnbetas_rpa, lamus_rpa, yerr=lamuserr_rpa, marker="o",linestyle="",fillstyle="none",markersize=2, label = "RPA",color=crpa)
plt.plot(lnbetas_rpa, lamus_rpa, marker="o",linestyle="",fillstyle="none",markersize=2, label = "RPA",color=crpa)
plt.plot(lnbetas_rpa, k_rpa*lnbetas_rpa+b_rpa, ":", lw=1,color=crpa, label=r"$\lambda=%.5f\times \log_{10}(\beta)%.5f$"%(k_rpa,b_rpa))

#plt.errorbar(lnbetas_ko, lamus_ko, yerr=lamuserr_ko, marker="v",linestyle="",fillstyle="none",markersize=2, label = "KO",color=cko)
plt.plot(lnbetas_ko, lamus_ko, marker="v",linestyle="",fillstyle="none",markersize=2, label = "KO",color=cko)
plt.plot(lnbetas_ko, k_ko*lnbetas_ko+b_ko, "--", lw=1,color=cko,label=r"$\lambda=%.5f\times \log_{10}(\beta)%.5f$"%(k_ko,b_ko))

plt.legend(fontsize="xx-small")
plt.xlabel(r"$log_{10}(\beta)$")
plt.ylabel(r"$\bar{\lambda}$")
plt.savefig("flow_rs"+"rs%2.2f_"%rs+"l%d"%nl+".pdf")
