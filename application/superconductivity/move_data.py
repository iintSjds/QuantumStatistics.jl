import os
import sys
size=10*128
dup=8
beta=10
r_s=3.0
channel=0
label=0
mom_sep = 0.2
if(len(sys.argv)==2):
	label=int(sys.argv[1])
for k in range(1,size+1):
        myCmd='cp {0}/*dat data/'.format(k)
	#myCmd='mv {0}/parameter* data/'.format(k)
        #myCmd='cp data/flow_{0}.dat {1}/'.format(k,k)
        #os.system(myCmd)
        #myCmd='cp data/delta_{0}.dat {1}/'.format(k,k)
        #myCmd='mv data/flow_{0}.dat {1}/'.format(k,k)
        os.system(myCmd)	
