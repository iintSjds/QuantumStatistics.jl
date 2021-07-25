import os
import sys
size=128*10
dup=8
beta=10
r_s=3.0
channel=0
label=0
mom_sep = 0.2
if(len(sys.argv)==2):
	label=int(sys.argv[1])
for k in range(1,size+1):
	myCmd='rm -rf {0}'.format(k)
	os.system(myCmd)	
