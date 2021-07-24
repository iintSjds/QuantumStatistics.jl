import os
import sys

dup=18
size0=dup*8 #
size=4*size0 #
print(size)
beta0=156.25*2**0.5
beta=156.25*2**0.5
r_s=2.0
channel=0
label=0
mom_sep0 = 0.0015625/3.0
mom_sep= mom_sep0
freq_sep = 2.0
if(len(sys.argv)==2):
	label=int(sys.argv[1])
for k in range(1,size+1):
	if(k%dup==1):
		beta=beta0
		#r_s=r_s+0.5
		mom_sep = mom_sep* 2.0**0.5
		#freq_sep = freq_sep /2.0**0.5
	if(k%size0==1):
		#r_s=r_s+0.5
		mom_sep = mom_sep0*r_s/8.0
		#freq_sep = 2.0
		channel=channel+1
	fname="parameter.jl"
	myCmd='mkdir {0}'.format(k)
	os.system(myCmd)
	#myCmd='cp .jl  {0}/'.format(k)
	#os.system(myCmd)
	#myCmd='cp eigen.jl  {0}/'.format(k)
	#os.system(myCmd)
	#myCmd='cp grid.jl  {0}/'.format(k)
	#os.system(myCmd)
	fo=open("{0}/".format(k)+fname,"w")
	fo.write(("module parameter\n"+
		"using StaticArrays, QuantumStatistics\n"+
		"const test_KL = false\n"+
		"const WID = %d\n"+
		"const me = 0.5\n"+	
		"const dim = 3\n"+
		"const spin = 2\n"+
		"const EPS = 1e-11\n"+
		"const rs = %f\n"+
		"const e0 = sqrt(rs*2.0/(9π/4.0)^(1.0/3))\n"+
		"const kF = 1.0\n"+
		"const EF = 1.0\n"+
		"const β = %f / kF^2\n"+
		"const mass2 = 0.0\n"+
		"const mass_Pi = 0\n"+
		"const mom_sep = %.10e\n"+
		"const mom_sep2 = 0.1\n"+
		"const freq_sep = 0.0\n"+
		"const channel = %d\n"+
		"const extK_grid = Grid.fermiKUL(kF, 10kF, 0.00001*sqrt(me^2/β/kF^2), 8,8)\n"+
		"const extT_grid = Grid.tauUL(β, 0.00001, 8,8)\n"+
		"const Nk = 16\n"+
		"const order = 4\n"+
		"const order_int = 16\n"+
		"const maxK = 10.0 * kF\n"+
		"const minK = 0.0000001 \n"+
		"for n in names(@__MODULE__; all=true)\n"+
		"	 if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)\n"+
		"		@eval export $n\n"+
		"	end\n"+
		"end\n"+
		"end")
		%(k, r_s, beta, mom_sep, channel))
	fo.close()
	beta = beta*(2**0.5)
	
