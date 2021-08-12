"""
Power method, damp interation and implicit renormalization
"""
#module eigensolver
#using QuantumStatistics
using QuantumStatistics: σx, σy, σz, σ0, Grid,FastMath, Utility, TwoPoint #, DLR, Spectral
using Lehmann
using LinearAlgebra
using Printf
#using Gaston
using Plots
using Statistics

srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

if !(@isdefined β)
    include(rundir*"/parameter.jl")
    using .parameter
end

include("eigen.jl")
include("grid.jl")

"""
Function Separation

Separate Gap function to low and high energy parts with given rule
"""

function Normalization(Δ0,Δ0_new, kgrid, qgrids)
    kpidx = 1 # panel index of the kgrid
    head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
    x = @view kgrid.grid[head:tail]
    w = @view kgrid.wgrid[head:tail]
    sum = 0
    for (qi, q) in enumerate(qgrids[1].grid)
        
        if q > kgrid.panel[kpidx + 1]
            # if q is larger than the end of the current panel, move k panel to the next panel
            while q > kgrid.panel[kpidx + 1]
                kpidx += 1
            end
            head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
            x = @view kgrid.grid[head:tail]
            w = @view kgrid.wgrid[head:tail]
            @assert kpidx <= kgrid.Np
        end
        wq = qgrids[1].wgrid[qi]
        Δ_int1 = @view Δ0[head:tail] # all F in the same kpidx-th K panel
        Δ_int2 = @view Δ0_new[head:tail] # all F in the same kpidx-th K panel
        DD1 = barycheb(order, q, Δ_int1, w, x) # the interpolation is independent with the panel length
        DD2 = barycheb(order, q, Δ_int2, w, x)
        #Δ0[ki] += bare(k, q) * FF * wq
        sum += DD1*DD2*wq                    
        @assert isfinite(sum) "fail normaliztion of Δ0"
    end
    return sum
end

function dlr_dot_kernal(ωn,ωm)
    A=0
    B=0
    if(abs(ωn-ωm)*β<1e-5)
        A =   -( β+(ωn-ωm)*β^2/2.0)*exp(-ωn*β)
    else
        A = (exp(-ωn*β)-exp(-ωm*β))/(ωn - ωm) 
    end
    if(abs(ωn+ωm)*β<1e-5)
        B = β-(ωn+ωm)*β^2/2.0
        
    else
        B = (1-exp(-(ωn+ωm)*β))/(ωn+ωm)
    end
    return 2*(A+B)
end

function dlr_dot(Δ,Δ_new,kgrid,qgrids, fdlr)
    coeff = DLR.tau2dlr(:acorr, Δ, fdlr, axis=2)
    coeff_new = DLR.tau2dlr(:acorr, Δ_new, fdlr, axis=2)
    result = zeros(Float64, length(Δ[:,1]))
    for i in 1:length(result)
        for (ωni,ωn) in enumerate(fdlr.ω)
            for  (ωmi,ωm) in enumerate(fdlr.ω)
                result[i] += coeff[i,ωni]*coeff_new[i,ωmi]*dlr_dot_kernal(ωn,ωm) 
            end
        end
    end

    kpidx = 1 # panel index of the kgrid
    head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
    x = @view kgrid.grid[head:tail]
    w = @view kgrid.wgrid[head:tail]
    sum = 0
    for (qi, q) in enumerate(qgrids[1].grid)
        
        if q > kgrid.panel[kpidx + 1]
            # if q is larger than the end of the current panel, move k panel to the next panel
            while q > kgrid.panel[kpidx + 1]
                kpidx += 1
            end
            head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
            x = @view kgrid.grid[head:tail]
            w = @view kgrid.wgrid[head:tail]
            @assert kpidx <= kgrid.Np
        end
        wq = qgrids[1].wgrid[qi]
        Δ_int1 = @view result[head:tail] # all F in the same kpidx-th K panel
        DD1 = barycheb(order, q, Δ_int1, w, x) # the interpolation is independent with the panel length
        #Δ0[ki] += bare(k, q) * FF * wq
        sum += DD1*wq                    
        @assert isfinite(sum) "fail normaliztion of Δ0"
    end
    return sum
end

function dlr_dot_freq(Δ,Δ_new,kgrid,qgrids, nfermi_grid)
    result = zeros(Float64, length(Δ[:,1]))
    for i in 1:length(result)
        for (ωni,ωn) in enumerate(nfermi_grid)
            
            result[i] += Δ[i,ωni]*Δ_new[i,ωni] 
          
        end
    end

    kpidx = 1 # panel index of the kgrid
    head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
    x = @view kgrid.grid[head:tail]
    w = @view kgrid.wgrid[head:tail]
    sum = 0
    for (qi, q) in enumerate(qgrids[1].grid)
        
        if q > kgrid.panel[kpidx + 1]
            # if q is larger than the end of the current panel, move k panel to the next panel
            while q > kgrid.panel[kpidx + 1]
                kpidx += 1
            end
            head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
            x = @view kgrid.grid[head:tail]
            w = @view kgrid.wgrid[head:tail]
            @assert kpidx <= kgrid.Np
        end
        wq = qgrids[1].wgrid[qi]
        Δ_int1 = @view result[head:tail] # all F in the same kpidx-th K panel
        DD1 = barycheb(order, q, Δ_int1, w, x) # the interpolation is independent with the panel length
        #Δ0[ki] += bare(k, q) * FF * wq
        sum += DD1*wq                    
        @assert isfinite(sum) "fail normaliztion of Δ0"
    end
    return sum/β
end


function Separation(delta0, delta, k::CompositeGrid,  fdlr)
    # cut=0.25
    # low=zeros(size(mom_grid)[1],size(freq_grid)[1])
    # high=zeros(size(mom_grid)[1],size(freq_grid)[1])
    # for i in 1:size(mom_grid)[1]
    #     for j in 1:size(freq_grid)[1]
    #         if(freq_grid[j]<cut)
    #             low[i,j]=1
    #         else
    #             high[i,j]=1
    #         end
    #     end
    # end
    #mom_sep = 0.1
    low=zeros(Float64, ( length(k.grid), fdlr.size))
    high=zeros(Float64, ( length(k.grid), fdlr.size))
    low0=zeros(Float64,  length(k.grid))
    high0=zeros(Float64,  length(k.grid))
    kindex_left = searchsortedfirst(k.panel, kF-mom_sep)
    kindex_right = searchsortedlast(k.panel, kF+mom_sep)
    #println("separation point:$(k.panel[kindex_left]),$(k.panel[kindex_right])")
    head = idx(kindex_left, 1, order)
    tail = idx(kindex_right, 1, order)-1
    # for i in 1:fdlr.size
    #     low[head:tail,i] .+= 1
    #     high[1:head-1,i] .+= 1
    #     high[tail+1:length(k.grid),i] .+= 1       
    # end
    # low0[head:tail] .+= 1
    # high0[1:head-1] .+= 1
    # high0[tail+1:length(k.grid)] .+= 1

    mom_width = mom_sep^2*0.5
    for i in 1:fdlr.size
        for (qi,q) in enumerate(k.grid)
            #if qi>=head && qi<tail
                #low[qi,i] = exp(-(q-kF)^2/mom_sep^2) 
                #low0[qi] = exp(-(q-kF)^2/mom_sep^2)
            #else
                #high[qi,i] = 1-exp(-(q-kF)^2/mom_sep^2) 
                #high0[qi] = 1- exp(-(q-kF)^2/mom_sep^2)                
            #end
	        low[qi,i] = (1.0 + exp(-mom_sep^2/mom_width))/(1.0+exp(((q-kF)^2-mom_sep^2)/mom_width))
                low0[qi] = (1.0 + exp(-mom_sep^2/mom_width))/(1.0+exp(((q-kF)^2-mom_sep^2)/mom_width))
            
                high[qi,i] = 1- (1.0 + exp(-mom_sep^2/mom_width))/(1.0+exp(((q-kF)^2-mom_sep^2)/mom_width))

                high0[qi] = 1- (1.0 + exp(-mom_sep^2/mom_width))/(1.0+exp(((q-kF)^2-mom_sep^2)/mom_width))

        end
    end
    



    #println(high0,low0,high0+low0)
    delta_0_low = low0 .* delta0
    delta_0_high = high0 .* delta0
    delta_low = low .* delta
    delta_high = high .* delta

    # for j in 1:length(mom_grid)
    #     for (ωi, ω) in enumerate(fdlr.ω)
    #         if(abs.(ω)<cut)
    #             low[ωi,j]=1
    #         else
    #             high[ωi,j]=1
    #         end
    #     end
    # end
    # coeff_low = coeff .* low
    # coeff_high = coeff .* high
    # delta_low =  DLR.dlr2matfreq(:acorr, coeff_low, fdlr, fdlr.n, axis=1)
    # delta_high = DLR.dlr2matfreq(:acorr, coeff_high, fdlr, fdlr.n, axis=1)

    # delta0_dum = zeros(ComplexF64, (fdlr.size, length(mom_grid)))
    # for j in 1:length(mom_grid)
    #     for (ωi, ω) in enumerate(fdlr.ω)
    #         delta0_dum[ωi,j]=delta0[j]
    #     end
    # end


    # coeff = DLR.matfreq2dlr(:acorr, delta0_dum, fdlr, axis=1)
    # println("coeff=",coeff[:,1])
    # coeff_low = coeff .* low
    # coeff_high = coeff .* high
    # delta_low = delta_low .+ DLR.dlr2matfreq(:acorr, coeff_low, fdlr, fdlr.n, axis=1)
    # delta_high = delta_high .+ DLR.dlr2matfreq(:acorr, coeff_high, fdlr, fdlr.n, axis=1)
    
    return delta_0_low, delta_0_high, delta_low, delta_high
end

function Separation_F(F_in,  k::CompositeGrid,  fdlr)
    # cut=0.25
    # low=zeros(size(mom_grid)[1],size(freq_grid)[1])
    # high=zeros(size(mom_grid)[1],size(freq_grid)[1])
    # for i in 1:size(mom_grid)[1]
    #     for j in 1:size(freq_grid)[1]
    #         if(freq_grid[j]<cut)
    #             low[i,j]=1
    #         else
    #             high[i,j]=1
    #         end
    #     end
    # end
    F=F_in
    
    coeff = DLR.tau2dlr(:acorr, F, fdlr, axis=2)
    #println(coeff[1:10,20])
    low=zeros(Float64, ( k.Np * k.order, fdlr.size))
    high=zeros(Float64, ( k.Np * k.order, fdlr.size))
    
    for (ωi, ω) in enumerate(fdlr.ω)
        for j in 1:(k.Np * k.order)
            if(abs.(ω*β)<freq_sep)
            #if(ωi<freq_sep_int)
                    
                low[j,ωi]=1
            else
                high[j,ωi]=1
            end
        end
    end
    coeff_low = coeff .* low
    coeff_high = coeff .* high
    F_low =  DLR.dlr2tau(:acorr, coeff_low, fdlr, fdlr.τ, axis=2)
    F_high = DLR.dlr2tau(:acorr, coeff_high, fdlr, fdlr.τ, axis=2)

    return real.(F_low), real.(F_high)
end



"""
Function Implicit_Renorm

    For given kernal, use implicit renormalization method to solve the eigenvalue

"""


function Implicit_Renorm(Δ, Δ_0 ,kernal, kernal_bare, kgrid, qgrids, fdlr )
    
    NN=100000
    rtol=1e-6
    Looptype=1
    n=0
    err=1.0 
    accm=0
    shift=5.0
    lamu0=-2.0
    lamu=0.0
    n_change=10  #steps of power method iteration in one complete loop
    n_change2=10+10 #total steps of one complete loop
    delta = Δ
    delta_0 = Δ_0
   
    #Separate Delta
    d_0_accm=zeros(Float64, length(kgrid.grid))
    d_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    delta_0_low, delta_0_high, delta_low, delta_high = Separation(delta_0, delta, kgrid, fdlr)
    F = zeros(Float64, (length(kgrid.grid), fdlr.size))
    while(n<NN && err>rtol)
        F=calcF(delta_0, delta, fdlr, kgrid)
       
        F_test = real(DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2))
        F_test =  real(DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2))
        println(maximum(abs.(F-F_test)))
        #println(abs.(kernal_freq_test-F_test)[kF_label,qF_label,:])
        n=n+1
        delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)
        #delta_new = real(DLR.tau2matfreq(:acorr, delta_new, fdlr, fdlr.n, axis=2))        
        
	 # if(n%n_change2==0)
   #          coeff_F = DLR.tau2dlr(:acorr, F, fdlr, axis=2)
   #          coeff_D = DLR.tau2dlr(:acorr, delta_new, fdlr, axis=2)
   #          @assert maximum(abs.(coeff_D))<1e16
   #          @assert maximum(abs.(coeff_F))<1e16
   #      end	



	delta_0_low_new, delta_0_high_new, delta_low_new, delta_high_new = Separation(delta_0_new, delta_new, kgrid, fdlr)
       
	if(Looptype==0)
            accm=accm+1
            d_0_accm = d_0_accm + delta_0_high_new 
            d_accm = d_accm + delta_high_new
            delta_0_high = d_0_accm ./ accm
            delta_high = d_accm ./ accm
        else
            # lamu = dot(delta_0_low_new, delta_0_low)
            # delta_0_low_new = delta_0_low_new+shift*delta_0_low
            # delta_low_new = delta_low_new+shift*delta_low
            # modulus = sqrt(dot(delta_0_low_new, delta_0_low_new))
            # delta_0_low = delta_0_low_new ./ modulus
            # delta_low = delta_low_new ./ modulus

            #lamu = dot(delta_low_new, delta_low)
            lamu = Normalization(delta_0_low, delta_0_low_new, kgrid, qgrids )
            delta_0_low_new = delta_0_low_new+shift*delta_0_low
            delta_low_new = delta_low_new+shift*delta_low
            modulus = Normalization(delta_0_low_new, delta_0_low_new, kgrid, qgrids)
            @assert modulus>0
            modulus = sqrt(modulus)
            delta_0_low = delta_0_low_new ./ modulus
            delta_low = delta_low_new ./ modulus
            #println(lamu)
        end
        delta_0 = delta_0_low .+ delta_0_high
        delta = delta_low .+ delta_high
        if(n%n_change2==n_change)
            Looptype=(Looptype+1)%2
        elseif(n%n_change2==0)
            accm = 0
            d_accm = d_accm .* 0
            d_0_accm = d_0_accm .* 0
            err=abs(lamu-lamu0)
            lamu0=lamu
            println(lamu)
            Looptype=(Looptype+1)%2
        end
        
    end

    #Separate F
    # F_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    # F=calcF(delta_0, delta, fdlr, kgrid)
    # F_low, F_high=Separation_F(F, kgrid, fdlr)
    # while(n<NN && err>rtol)
    #     n=n+1
    #     delta_0_new, delta_new = calcΔ(F, kernal, fdlr , kgrid, qgrids)./(-4*π*π)
    #     F_new=calcF(delta_0_new, delta_new, fdlr, kgrid)
    #     F_low_new, F_high_new = Separation_F(F_new, kgrid, fdlr)
    #     println(Looptype)
    #     if(Looptype==0)
    #         accm=accm+1
    #         F_accm=F_accm+F_high_new
    #         F_high=F_accm./accm
    #     else
    #         lamu=dot(F_low, F_low_new)
    #         F_low_new=F_low_new+shift*F_low
    #         modulus=sqrt(dot(F_low_new, F_low_new))
    #         F_low=F_low_new/modulus
    #         println(lamu)
    #     end
    #     F = F_low+F_high
    #     if(n%n_change2==n_change)
    #         Looptype=(Looptype+1)%2
    #     elseif(n%n_change2==0)
    #         accm=0
    #         F_accm=0*F_accm
    #         err=abs(lamu-lamu0)
    #         lamu0=lamu
    #         println(lamu)
    #         Looptype=(Looptype+1)%2
    #     end
    # end
    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "a")
    @printf(f, "%32.17g\n", lamu)
    close(f)
    #F=calcF(delta_0, delta, fdlr, kgrid)
    #delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)


    return delta_0_low, delta_0_high, delta_low, delta_high, F, lamu
end

function Implicit_Renorm_Test_Mom(kernal, kernal_bare, kgrid, qgrids, fdlr, kgrid2, qgrids2 )
    NN=100000
    rtol=1e-6
    Looptype=1
    n=0
    err=1.0 
    accm=0
    shift=5.0
    lamu0=-2.0
    lamu=0.0
    n_change=10  #steps of power method iteration in one complete loop
    n_change2=10+10 #total steps of one complete loop

    delta = zeros(Float64, (length(kgrid.grid), fdlr.size))
    for (ki, k) in enumerate(kgrid.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            delta[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
        end
    end

    delta_0 = zeros(Float64, length(kgrid.grid)) .+ 1.0
    #Separate Delta
    d_0_accm=zeros(Float64, length(kgrid.grid))
    d_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    delta_0_low, delta_0_high, delta_low, delta_high = Separation(delta_0, delta, kgrid, fdlr)
    F = zeros(Float64, (length(kgrid.grid), fdlr.size))
    while(n<NN && err>rtol)
        F=calcF(delta_0, delta, fdlr, kgrid)
        n=n+1
        delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)
        #delta_new = real(DLR.tau2matfreq(:acorr, delta_new, fdlr, fdlr.n, axis=2))        
        delta_0_low_new, delta_0_high_new, delta_low_new, delta_high_new = Separation(delta_0_new, delta_new, kgrid, fdlr)

        # test mom interp for F
        diffF=testMomInterp(F, fdlr, kgrid, qgrids, kgrid2, qgrids2)
        val, loc = findmax(abs.(diffF))
        println("Finterperr:",[kgrid.grid[loc[1]], fdlr.τ[loc[2]], F[loc[1],loc[2]]], "\t", val)
        println("Fmax:",maximum(F), "\t avgerr", mean(abs.(diffF)))
        p = plot(kgrid.grid, F[:,loc[2]]) #, xlim=(xMin,xMax), ylim=(yMin, yMax))
        plot!(p, kgrid.grid, (F+diffF)[:,loc[2]]) #, xlim=(xMin,xMax), ylim=(yMin, yMax))
        plot!(p, kgrid.grid[loc[1]:loc[1]], F[loc[1]:loc[1], loc[2]]) #, xlim=(xMin,xMax), ylim=(yMin, yMax))
        display(p)
        readline()


        if(Looptype==0)
            accm=accm+1
            d_0_accm = d_0_accm + delta_0_high_new 
            d_accm = d_accm + delta_high_new
            delta_0_high = d_0_accm ./ accm
            delta_high = d_accm ./ accm
        else
            # lamu = dot(delta_0_low_new, delta_0_low)
            # delta_0_low_new = delta_0_low_new+shift*delta_0_low
            # delta_low_new = delta_low_new+shift*delta_low
            # modulus = sqrt(dot(delta_0_low_new, delta_0_low_new))
            # delta_0_low = delta_0_low_new ./ modulus
            # delta_low = delta_low_new ./ modulus

            #lamu = dot(delta_low_new, delta_low)
            lamu = Normalization(delta_0_low, delta_0_low_new, kgrid )
            println(lamu)
            delta_0_low_new = delta_0_low_new+shift*delta_0_low
            delta_low_new = delta_low_new+shift*delta_low
            modulus = Normalization(delta_0_low_new, delta_0_low_new, kgrid)
            @assert modulus>0
            modulus = sqrt(modulus)
            delta_0_low = delta_0_low_new ./ modulus
            delta_low = delta_low_new ./ modulus
            #println(lamu)
        end
        delta_0 = delta_0_low .+ delta_0_high
        delta = delta_low .+ delta_high
        if(n%n_change2==n_change)
            Looptype=(Looptype+1)%2
        elseif(n%n_change2==0)
            accm = 0
            d_accm = d_accm .* 0
            d_0_accm = d_0_accm .* 0
            err=abs(lamu-lamu0)
            lamu0=lamu
            println(lamu)
            Looptype=(Looptype+1)%2
        end
        
    end

    #Separate F
    # F_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    # F=calcF(delta_0, delta, fdlr, kgrid)
    # F_low, F_high=Separation_F(F, kgrid, fdlr)
    # while(n<NN && err>rtol)
    #     n=n+1
    #     delta_0_new, delta_new = calcΔ(F, kernal, fdlr , kgrid, qgrids)./(-4*π*π)
    #     F_new=calcF(delta_0_new, delta_new, fdlr, kgrid)
    #     F_low_new, F_high_new = Separation_F(F_new, kgrid, fdlr)
    #     println(Looptype)
    #     if(Looptype==0)
    #         accm=accm+1
    #         F_accm=F_accm+F_high_new
    #         F_high=F_accm./accm
    #     else
    #         lamu=dot(F_low, F_low_new)
    #         F_low_new=F_low_new+shift*F_low
    #         modulus=sqrt(dot(F_low_new, F_low_new))
    #         F_low=F_low_new/modulus
    #         println(lamu)
    #     end
    #     F = F_low+F_high
    #     if(n%n_change2==n_change)
    #         Looptype=(Looptype+1)%2
    #     elseif(n%n_change2==0)
    #         accm=0
    #         F_accm=0*F_accm
    #         err=abs(lamu-lamu0)
    #         lamu0=lamu
    #         println(lamu)
    #         Looptype=(Looptype+1)%2
    #     end
    # end
    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "w")
    @printf(f, "%32.17g\n", lamu)
    #F=calcF(delta_0, delta, fdlr, kgrid)
    #delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)


    return delta_0, delta, F
end





function Explicit_Solver(kernal, kernal_bare, kgrid, qgrids, fdlr, bdlr )
    NN=10000
    rtol=1e-6
    n=0
    modulus= 1.0
    err=1.0 
    shift=5.0
    lamu0=-2.0
    lamu0_2= -2.0
    lamu=0.0
    lamu2=0.0
    #kernal_bare, kernal = dH1_freq(kgrid, qgrids, bdlr, fdlr)
    #kernal = dH1_tau(kgrid, qgrids, fdlr)
    #kernal_double = dH1_tau(kgrid_double, qgrids_double, fdlr)
    #kernal_2 = dH1_tau(kgrid, qgrids, fdlr2)
    delta = zeros(Float64, (length(kgrid.grid), fdlr.size))
    # for (ki, k) in enumerate(kgrid.grid)
    #     ω = k^2 / (2me) - EF
    #     for (ni, n) in enumerate(fdlr.n)
    #         np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
    #         nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
    #         # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    #         delta[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    #     end
    # end
    # delta = real(DLR.matfreq2tau(:acorr, delta, fdlr, fdlr.τ, axis=2))
    delta_0 = zeros(Float64, length(kgrid.grid)) .+ 1.0
    delta_0_new=zeros(Float64, length(kgrid.grid))
    delta_new=zeros(Float64, (length(kgrid.grid), fdlr.size))
    F = zeros(Float64, (length(kgrid.grid), fdlr.size))

    # Δ_2 = zeros(Float64, (length(kgrid.grid), fdlr2.size))
    # Δ0_2 = zeros(Float64, length(kgrid.grid)) #.+ 1.0
    # Δ0_2_new=zeros(Float64, length(kgrid.grid))
    # Δ_2_new=zeros(Float64, (length(kgrid.grid), fdlr2.size))
    # Δ_2_int = zeros(Float64, (length(kgrid.grid), fdlr.size))

    # for (ki, k) in enumerate(kgrid.grid)
    #     ω = k^2 / (2me) - EF
    #     for (ni, n) in enumerate(fdlr2.n)
    #         np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
    #         nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
    #         # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    #         Δ_2[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)

    #     end
    # end
    # Δ_2 = real(DLR.matfreq2tau(:acorr, Δ_2, fdlr2, fdlr2.τ, axis=2))

    # for (ki, k) in enumerate(kgrid.grid)
    #     ω = k^2 / (2me) - EF
    #     for (ni, n) in enumerate(fdlr.n)
    #         np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
    #         nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
    #         # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    #         Δ_2_int[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    #     end
    # end
    # Δ_2_int = real(DLR.matfreq2tau(:acorr, Δ_2_int, fdlr, fdlr.τ, axis=2))
    # Δ_init = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
    # Δ_init = DLR.dlr2tau(:acorr, Δ_init, fdlr2, [1.0e-12, β-1.0e-12] , axis=2)

    # println("Δ_init=$(Δ_init[1,:] )")
    #Separate Delta
   
    while(n<NN && err>rtol)
        #delta = real(DLR.tau2matfreq(:acorr, delta, fdlr, fdlr.n, axis=2))        
        F=calcF(delta_0, delta, fdlr, kgrid)
        #pic = plot(fdlr.τ[1:30], F[kF_label,1:30]) 
        # #pic = plot!(L_fdlr.n[1:10], L_F[kF_label,1:10])
        # #pic = plot(fdlr.n[1:10], delta[kF_label,1:10]) 
        # #pic = plot!(L_fdlr.n[1:10], L_delta[kF_label,1:10])
        
        #display(pic)
        #readline()
        # outFileName = rundir*"/F_$(WID).dat"
        # f = open(outFileName, "w")
        # for (ki, k) in enumerate(kgrid.grid)
        #     for (ni, n) in enumerate(fdlr.n)
        #         @printf(f, "%32.17g  %32.17g\n",fdlr.τ[:], F[kF_label,:] )
        #     end
        # end

        n=n+1
        #p = plot(fdlr.τ[:], F[kF_label,:]) #, xlim=(xMin,xMax), ylim=(yMin, yMax))
        #p = plot!(fdlr.τ[:], qgrids[1].grid, kernal_int[1, :, bdlr.size÷2+1])
        #display(p)
        #readline()
        #println("F: $F[kF_label, :]")
        #readline()
        #test dlr err
        # F_test =  DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # F_test =  DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2)
        
        # println("err_F_real=",maximum(abs.(real.(F_test) - F)))
        # println("err_F_imag=", maximum(abs.(imag.(F_test) )))
        # println("err_F_ratio=",maximum(abs.(real.(F_test) - F)./abs.(F)))
        delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)
        #pic = plot(fdlr.τ[1:30], delta_new[kF_label,1:30]) 
        # #pic = plot!(L_fdlr.n[1:10], L_F[kF_label,1:10])
        # #pic = plot(fdlr.n[1:10], delta[kF_label,1:10]) 
        # #pic = plot!(L_fdlr.n[1:10], L_delta[kF_label,1:10])
        
        #display(pic)
        #readline()
        delta_freq = real(DLR.tau2matfreq(:acorr, delta_new, fdlr, fdlr.n, axis=2))
        #p = plot(fdlr.n[:], delta_freq[kF_label,:]) #, xlim=(xMin,xMax), ylim=(yMin, yMax))
        #p = plot(kgrid.grid, delta_freq[:,1]) #, xlim=(xMin,xMax), ylim=(yMin, yMax))
        
        #p = plot!(fdlr.τ[:], qgrids[1].grid, kernal_int[1, :, bdlr.size÷2+1])
        #display(p)
        #readline()
        # outFileName = rundir*"/delta_$(WID).dat"
        # f = open(outFileName, "w")
        # for (ki, k) in enumerate(kgrid.grid)
        #     for (ni, n) in enumerate(fdlr.n)
        #         @printf(f, "%32.17g  %32.17g\n",fdlr.n[ni], Δ_freq[ki, ni] + Δ0_final[ki])
        #     end
        # end



        #println("Delta0: $delta_0_new[:]")
        #readline()
        #println("Deltta: $delta_freq[kF_label, :]")
        #readline()
        #delta_0_low_new, delta_0_high_new, delta_low_new, delta_high_new = Separation(delta_0_new, delta_new, kgrid, fdlr)
        lamu = dot(delta_new, delta)
        #lamu = Normalization(delta_0, delta_0_new, kgrid )
        delta_0_new = delta_0_new+shift*delta_0
        delta_new = delta_new+shift*delta
        #modulus = Normalization(delta_0_new, delta_0_new, kgrid)
        modulus = sqrt(dot(delta_new, delta_new))
        delta_0 = delta_0_new ./ modulus
        delta = delta_new ./ modulus
        err=abs(lamu-lamu0)
        lamu0=lamu
        println(lamu)
    
        # # F_2 = calcF(Δ0_2, Δ_2, fdlr2, kgrid)

        # # F_2_int = DLR.tau2dlr(:acorr, F_2, fdlr2, axis=2)
        # # F = DLR.dlr2tau(:acorr, F_2_int, fdlr2, fdlr.τ , axis=2)


        # # # test dlr err
        # # # F_test =  DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # # # F_test =  DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2)
       
        # # # println("err_F_real=",maximum(abs.(real.(F_test) - F)))
        # # # println("err_F_imag=", maximum(abs.(imag.(F_test) )))
        # # # println("err_F_ratio=",maximum(abs.(real.(F_test) - F)./abs.(F)))
        # # delta_0_new, delta_new = calcΔ(F, kernal, fdlr, kgrid, qgrids)./(-4*π*π)
        
        # # Δ0_2_new, Δ_2_new = calcΔ(F_2, kernal_2, fdlr2 ,kgrid, qgrids)./(-4*π*π)

        # # Δ_freq = DLR.tau2matfreq(:acorr, delta_new, fdlr, fdlr.n, axis=2)
        # # Δ_freq_2 = DLR.tau2matfreq(:acorr, Δ_2_new, fdlr2, fdlr.n, axis=2)
        # # Δ_2_int = DLR.tau2dlr(:acorr, Δ_2_new, fdlr2, axis=2)
        # # Δ_2_int = DLR.dlr2tau(:acorr, Δ_2_int, fdlr2, fdlr.τ , axis=2)
        # # err0 = maximum(abs.(Δ0_2_new-delta_0_new))
        # # err = maximum(abs.(Δ_2_int-delta_new))
        # # err_freq = maximum(abs.(Δ_freq_2-Δ_freq))
        # # ind=findmax(abs.(Δ0_2_new-delta_0_new))[2]

        # # println("err0: $(err0), err: $(err),err_freq: $(err_freq) ,mom: $(kgrid.grid[ind[1]])")

        # #compare F
        # F=calcF(Δ0_2, Δ_2_int, fdlr, kgrid)
        # F_2 = calcF(Δ0_2, Δ_2, fdlr2, kgrid)
        
        # F_2_int = DLR.tau2dlr(:acorr, F_2, fdlr2, axis=2)
        # F_2_int = DLR.dlr2tau(:acorr, F_2_int, fdlr2, fdlr.τ , axis=2)

        # F_freq = DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # F_2_freq = DLR.tau2matfreq(:acorr, F_2, fdlr2, fdlr.n, axis=2)
        # err0 = maximum(abs.(F_2_int-F))
        # err = maximum(abs.(F_2_freq-F_freq))
        # ind=findmax(abs.(F_2_int-F))[2]

        # println("tau_err: $(err0), freq_err: $(err) ,mom: $(kgrid.grid[ind[1]]), freq: $(fdlr.n[ind[2]])")
        # # p = plot(fdlr.τ, F_2_int[ind[1],:] )
        # # p = plot!(fdlr.τ, F[ind[1],:])
        # # # test dlr err
        # # # F_test =  DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # # # F_test =  DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2)
        # # display(p)
        # # readline()
        # # println("err_F_real=",maximum(abs.(real.(F_test) - F)))
        # # println("err_F_imag=", maximum(abs.(imag.(F_test) )))
        # # println("err_F_ratio=",maximum(abs.(real.(F_test) - F)./abs.(F)))
        # Δ0_2_new, Δ_2_new = calcΔ(F_2, kernal_2, fdlr2 ,kgrid, qgrids)./(-4*π*π)  
       
 


        # #test interpolation err
        # #lamu = F_new[kf_label,1]/F[kf_label,1]
        # # if(n%10==0)
        # #     F_double =  zeros(Float64, (length(kgrid_double.grid), fdlr.size))
        # #     for τi in 1:fdlr.size
        # #         F_double[:, τi] = interpolate(F[:, τi], kgrid, kgrid_double.grid)
        # #     end
        # #     delta_0_double, delta_double = calcΔ(F_double, kernal_double, fdlr, kgrid_double, qgrids_double)./(-4*π*π)
        # #     delta_0_fine = interpolate(delta_0_double, kgrid_double, kgrid.grid)
        # #     printstyled("Max Err for Δ0 interpolation: ", maximum(abs.(delta_0_new - delta_0_fine)), "\n", color=:red)
        # # end
        

        # # lamu = dot(delta, delta_new)
        # # delta_0_new=delta_0_new+shift*delta_0
        # # delta_new=delta_new+shift*delta
        # # modulus=sqrt(dot(delta_new, delta_new))
        # # #modulus = abs(F_new[kf_label,1])
        # # delta_0 = delta_0_new/modulus
        # # delta = delta_new/modulus
        # # println("$(modulus), $(maximum(abs.(Δ_2)))")
        # # err=abs(lamu-lamu0)
        # # lamu0=lamu
        # # println(lamu)




        # lamu2 = dot(Δ_2, Δ_2_new)
        # Δ0_2_new=Δ0_2_new+shift*Δ0_2
        # Δ_2_new=Δ_2_new+shift*Δ_2
       
        # #modulus = abs(F_new[kf_label,1])
        # err0 = maximum(abs.(abs.(Δ_2_new)-abs.(Δ_2*modulus)))
        # err = maximum(abs.(abs.(Δ0_2_new)-abs.(Δ0_2*modulus)))
        # ind=findmax(abs.(abs.(Δ_2_new)-abs.(Δ_2*modulus)))[2]
        # ind0=findmax(abs.(abs.(Δ0_2_new)-abs.(Δ0_2*modulus)))[2]
        
        # modulus=sqrt(dot(Δ_2_new, Δ_2_new))
        # println("Δ change dynamic: $(err0), static: $(err) ,mom_dy: $(kgrid.grid[ind[1]]), τ_dy: $(fdlr2.τ[ind[2]]), mom_st: $(kgrid.grid[ind0[1]]), value_st:$(Δ0_2_new[ind0[1]]), value_st_old:$(Δ0_2[ind0[1]]*modulus)")
        # Δ0_2 = Δ0_2_new/modulus
        # Δ_2 = Δ_2_new/modulus
        # #println("$(modulus), $(maximum(abs.(Δ_2)))")
        # lamu0_2=lamu2
        # println(lamu2)
        
        # # Δ_2_int_2 = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
        # # Δ_2_int_2 = DLR.dlr2tau(:acorr, Δ_2_int_2, fdlr2, fdlr.τ , axis=2)

        # # err0 = maximum(abs.(Δ0_2-delta_0))
        # # err = maximum(abs.(Δ_2_int_2-delta))
        # # #err_freq = maximum(abs.(Δ_freq_2-Δ_freq))
        # # ind=findmax(abs.(Δ0_2-delta_0))[2]

        # # println("err0: $(err0), err: $(err),err_freq: $(err_freq) ,mom: $(kgrid.grid[ind[1]])")
        
        # Δ_2_int = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
        # Δ_2_int = DLR.dlr2tau(:acorr, Δ_2_int, fdlr2, fdlr.τ , axis=2)


    end

    #Separate F
    # F=calcF(delta_0, delta, fdlr, kgrid)
    # while(n<NN)# && err>rtol)
    #     n=n+1
    #     delta_0_new, delta_new = calcΔ(F, kernal, fdlr, kgrid, qgrids)./(-4*π*π)
    #     F_new=calcF(delta_0_new, delta_new, fdlr, kgrid)
    #     #lamu = F_new[kf_label,1]/F[kf_label,1]
    #     lamu = dot(F, F_new)
    #     F_new=F_new+shift*F
    #     modulus =sqrt(dot(F_new, F_new))
    #     #modulus = abs(F_new[kf_label,1])
    #     F=F_new/modulus
    #     err=abs(lamu-lamu0)
    #     lamu0=lamu
    #     println(lamu)
    # end
    #return delta_0_new, delta_new
    return delta_0, delta, F
end

function Explicit_Solver_err(kernal, kernal2, kernal_bare, kgrid, qgrids, fdlr, fdlr2,  bdlr )
    NN=10000
    rtol=1e-6
    n=0
    modulus= 1.0
    err=1.0 
    shift=5.0
    lamu0=-2.0
    lamu0_2= -2.0
    lamu=0.0
    lamu2=0.0
    #kernal_bare, kernal = dH1_freq(kgrid, qgrids, bdlr, fdlr)
    #kernal = dH1_tau(kgrid, qgrids, fdlr)
    #kernal_double = dH1_tau(kgrid_double, qgrids_double, fdlr)
    #kernal_2 = dH1_tau(kgrid, qgrids, fdlr2)
    delta = zeros(Float64, (length(kgrid.grid), fdlr.size))
    for (ki, k) in enumerate(kgrid.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            delta[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
        end
    end
    delta = real(DLR.matfreq2tau(:acorr, delta, fdlr, fdlr.τ, axis=2))
    delta_0 = zeros(Float64, length(kgrid.grid)) #.+ 1.0
    delta_0_new=zeros(Float64, length(kgrid.grid))
    delta_new=zeros(Float64, (length(kgrid.grid), fdlr.size))
    F = zeros(Float64, (length(kgrid.grid), fdlr.size))

    Δ_2 = zeros(Float64, (length(kgrid.grid), fdlr2.size))
    Δ0_2 = zeros(Float64, length(kgrid.grid)) #.+ 1.0
    Δ0_2_new=zeros(Float64, length(kgrid.grid))
    Δ_2_new=zeros(Float64, (length(kgrid.grid), fdlr2.size))
    Δ_2_int = zeros(Float64, (length(kgrid.grid), fdlr.size))

    for (ki, k) in enumerate(kgrid.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr2.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            Δ_2[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)

        end
    end
    Δ_2 = real(DLR.matfreq2tau(:acorr, Δ_2, fdlr2, fdlr2.τ, axis=2))

    for (ki, k) in enumerate(kgrid.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            Δ_2_int[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
        end
    end
    Δ_2_int = real(DLR.matfreq2tau(:acorr, Δ_2_int, fdlr, fdlr.τ, axis=2))
    Δ_init = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
    Δ_init = DLR.dlr2tau(:acorr, Δ_init, fdlr2, [1.0e-12, β-1.0e-12] , axis=2)

    # println("Δ_init=$(Δ_init[1,:] )")
    #Separate Delta
   
    while(n<NN)
        # #delta = real(DLR.tau2matfreq(:acorr, delta, fdlr, fdlr.n, axis=2))        
        # F=calcF(delta_0, delta, fdlr, kgrid)
        # n=n+1
        # #test dlr err
        # # F_test =  DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # # F_test =  DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2)
        
        # # println("err_F_real=",maximum(abs.(real.(F_test) - F)))
        # # println("err_F_imag=", maximum(abs.(imag.(F_test) )))
        # # println("err_F_ratio=",maximum(abs.(real.(F_test) - F)./abs.(F)))
        # delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)
        # #delta_0_low_new, delta_0_high_new, delta_low_new, delta_high_new = Separation(delta_0_new, delta_new, kgrid, fdlr)
        # lamu = dot(delta_new, delta)
        # #lamu = Normalization(delta_0, delta_0_new, kgrid )
        # delta_0_new = delta_0_new+shift*delta_0
        # delta_new = delta_new+shift*delta
        # #modulus = Normalization(delta_0_new, delta_0_new, kgrid)
        # modulus = sqrt(dot(delta_new, delta_new))
        # delta_0 = delta_0_new ./ modulus
        # delta = delta_new ./ modulus
        # err=abs(lamu-lamu0)
        # lamu0=lamu
        # println(lamu)
    
        F_2 = calcF(Δ0_2, Δ_2, fdlr2, kgrid)

        F_2_int = DLR.tau2dlr(:acorr, F_2, fdlr2, axis=2)
        F = DLR.dlr2tau(:acorr, F_2_int, fdlr2, fdlr.τ , axis=2)


        # test dlr err
        # F_test =  DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # F_test =  DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2)
       
        # println("err_F_real=",maximum(abs.(real.(F_test) - F)))
        # println("err_F_imag=", maximum(abs.(imag.(F_test) )))
        # println("err_F_ratio=",maximum(abs.(real.(F_test) - F)./abs.(F)))
        delta_0_new, delta_new = calcΔ(F, kernal,kernal_bare, fdlr, kgrid, qgrids)./(-4*π*π)
        
        Δ0_2_new, Δ_2_new = calcΔ(F_2, kernal2,kernal_bare, fdlr2 ,kgrid, qgrids)./(-4*π*π)

        Δ_freq = DLR.tau2matfreq(:acorr, delta_new, fdlr, fdlr.n, axis=2)
        Δ_freq_2 = DLR.tau2matfreq(:acorr, Δ_2_new, fdlr2, fdlr.n, axis=2)
        Δ_2_int = DLR.tau2dlr(:acorr, Δ_2_new, fdlr2, axis=2)
        Δ_2_int = DLR.dlr2tau(:acorr, Δ_2_int, fdlr2, fdlr.τ , axis=2)
        err0 = maximum(abs.(Δ0_2_new-delta_0_new))
        err = maximum(abs.(Δ_2_int-delta_new))
        err_freq = maximum(abs.(Δ_freq_2-Δ_freq))
        ind=findmax(abs.(Δ0_2_new-delta_0_new))[2]

        println("err0: $(err0), err: $(err),err_freq: $(err_freq) ,mom: $(kgrid.grid[ind[1]])")

        # #compare F
        # F=calcF(Δ0_2, Δ_2_int, fdlr, kgrid)
        # F_2 = calcF(Δ0_2, Δ_2, fdlr2, kgrid)
        
        # F_2_int = DLR.tau2dlr(:acorr, F_2, fdlr2, axis=2)
        # F_2_int = DLR.dlr2tau(:acorr, F_2_int, fdlr2, fdlr.τ , axis=2)

        # F_freq = DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # F_2_freq = DLR.tau2matfreq(:acorr, F_2, fdlr2, fdlr.n, axis=2)
        # err0 = maximum(abs.(F_2_int-F))
        # err = maximum(abs.(F_2_freq-F_freq))
        # ind=findmax(abs.(F_2_int-F))[2]

        # println("tau_err: $(err0), freq_err: $(err) ,mom: $(kgrid.grid[ind[1]]), freq: $(fdlr.n[ind[2]])")
        # # p = plot(fdlr.τ, F_2_int[ind[1],:] )
        # # p = plot!(fdlr.τ, F[ind[1],:])
        # # # test dlr err
        # # # F_test =  DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # # # F_test =  DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2)
        # # display(p)
        # # readline()
        # # println("err_F_real=",maximum(abs.(real.(F_test) - F)))
        # # println("err_F_imag=", maximum(abs.(imag.(F_test) )))
        # # println("err_F_ratio=",maximum(abs.(real.(F_test) - F)./abs.(F)))
        # Δ0_2_new, Δ_2_new = calcΔ(F_2, kernal_2, fdlr2 ,kgrid, qgrids)./(-4*π*π)  
       
 


        # #test interpolation err
        # #lamu = F_new[kf_label,1]/F[kf_label,1]
        # # if(n%10==0)
        # #     F_double =  zeros(Float64, (length(kgrid_double.grid), fdlr.size))
        # #     for τi in 1:fdlr.size
        # #         F_double[:, τi] = interpolate(F[:, τi], kgrid, kgrid_double.grid)
        # #     end
        # #     delta_0_double, delta_double = calcΔ(F_double, kernal_double, fdlr, kgrid_double, qgrids_double)./(-4*π*π)
        # #     delta_0_fine = interpolate(delta_0_double, kgrid_double, kgrid.grid)
        # #     printstyled("Max Err for Δ0 interpolation: ", maximum(abs.(delta_0_new - delta_0_fine)), "\n", color=:red)
        # # end
        

        # # lamu = dot(delta, delta_new)
        # # delta_0_new=delta_0_new+shift*delta_0
        # # delta_new=delta_new+shift*delta
        # # modulus=sqrt(dot(delta_new, delta_new))
        # # #modulus = abs(F_new[kf_label,1])
        # # delta_0 = delta_0_new/modulus
        # # delta = delta_new/modulus
        # # println("$(modulus), $(maximum(abs.(Δ_2)))")
        # # err=abs(lamu-lamu0)
        # # lamu0=lamu
        # # println(lamu)




        lamu2 = dot(Δ_2, Δ_2_new)
        Δ0_2_new=Δ0_2_new+shift*Δ0_2
        Δ_2_new=Δ_2_new+shift*Δ_2
       
        #modulus = abs(F_new[kf_label,1])
        # err0 = maximum(abs.(abs.(Δ_2_new)-abs.(Δ_2*modulus)))
        # err = maximum(abs.(abs.(Δ0_2_new)-abs.(Δ0_2*modulus)))
        # ind=findmax(abs.(abs.(Δ_2_new)-abs.(Δ_2*modulus)))[2]
        # ind0=findmax(abs.(abs.(Δ0_2_new)-abs.(Δ0_2*modulus)))[2]
        
        modulus=sqrt(dot(Δ_2_new, Δ_2_new))
        #println("Δ change dynamic: $(err0), static: $(err) ,mom_dy: $(kgrid.grid[ind[1]]), τ_dy: $(fdlr2.τ[ind[2]]), mom_st: $(kgrid.grid[ind0[1]]), value_st:$(Δ0_2_new[ind0[1]]), value_st_old:$(Δ0_2[ind0[1]]*modulus)")
        Δ0_2 = Δ0_2_new/modulus
        Δ_2 = Δ_2_new/modulus
        #println("$(modulus), $(maximum(abs.(Δ_2)))")
        lamu0_2=lamu2
        println(lamu2)
        
        # # Δ_2_int_2 = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
        # # Δ_2_int_2 = DLR.dlr2tau(:acorr, Δ_2_int_2, fdlr2, fdlr.τ , axis=2)

        # # err0 = maximum(abs.(Δ0_2-delta_0))
        # # err = maximum(abs.(Δ_2_int_2-delta))
        # # #err_freq = maximum(abs.(Δ_freq_2-Δ_freq))
        # # ind=findmax(abs.(Δ0_2-delta_0))[2]

        # # println("err0: $(err0), err: $(err),err_freq: $(err_freq) ,mom: $(kgrid.grid[ind[1]])")
        
        # Δ_2_int = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
        # Δ_2_int = DLR.dlr2tau(:acorr, Δ_2_int, fdlr2, fdlr.τ , axis=2)


    end

    #Separate F
    # F=calcF(delta_0, delta, fdlr, kgrid)
    # while(n<NN)# && err>rtol)
    #     n=n+1
    #     delta_0_new, delta_new = calcΔ(F, kernal, fdlr, kgrid, qgrids)./(-4*π*π)
    #     F_new=calcF(delta_0_new, delta_new, fdlr, kgrid)
    #     #lamu = F_new[kf_label,1]/F[kf_label,1]
    #     lamu = dot(F, F_new)
    #     F_new=F_new+shift*F
    #     modulus =sqrt(dot(F_new, F_new))
    #     #modulus = abs(F_new[kf_label,1])
    #     F=F_new/modulus
    #     err=abs(lamu-lamu0)
    #     lamu0=lamu
    #     println(lamu)
    # end
    #return delta_0_new, delta_new
    return delta_0, delta, F
end



function test_Lehmann(delta_0, delta,L_delta_0, L_delta)

    shift = 5.0
    for i in 1:10
        F=calcF(delta_0, delta, fdlr, kgrid)
        L_F = L_calcF(L_delta_0, L_delta, L_fdlr, kgrid)

        F_freq =  real.(DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2))
        L_F_freq =  real.(Lehmann.DLR.tau2matfreq(:acorr, L_F, L_fdlr, fdlr.n, axis=2))
        println(maximum(abs.(F_freq-L_F_freq)))
        #pic = plot(fdlr.τ, F[kF_label,:]) 
        #pic = plot!(L_fdlr.τ, L_F[kF_label,:])

        #pic = plot(fdlr.n, F_freq[kF_label,:]) 
        #pic = plot!(L_fdlr.n, L_F_freq[kF_label,:])
        #pic = plot(fdlr.n[1:10], delta[kF_label,1:10]) 
        #pic = plot!(L_fdlr.n[1:10], L_delta[kF_label,1:10])
        
        #display(pic)
        #readline()
        delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)
        L_delta_0_new, L_delta_new =  L_calcΔ(L_F, L_kernal, L_kernal_bare, L_fdlr , kgrid, qgrids)./(-4*π*π)

        
        d_freq =  real.(DLR.tau2matfreq(:acorr, delta_new, fdlr, fdlr.n, axis=2))
        L_d_freq =  real.(Lehmann.DLR.tau2matfreq(:acorr, L_delta_new, L_fdlr, fdlr.n, axis=2))
        ind=findmax(abs.(d_freq-L_d_freq))[2]
        println("max gg err:$(maximum(abs.(d_freq-L_d_freq))), $(kgrid.grid[ind[1]]), $(fdlr.n[ind[2]])")
        coeff_F = DLR.tau2dlr(:acorr, F, fdlr, axis=2)
        coeff_LF = Lehmann.DLR.tau2dlr(:acorr, L_F, L_fdlr, axis=2)
        coeff_D = DLR.tau2dlr(:acorr, delta_new, fdlr, axis=2)
        coeff_LD = Lehmann.DLR.tau2dlr(:acorr, L_delta_new, L_fdlr, axis=2)
        
        println("Fdlr coeff: old $(maximum(abs.(coeff_F))), new $(maximum(abs.(coeff_LF)))")
        println("Ddlr coeff: old $(maximum(abs.(coeff_D))), new $(maximum(abs.(coeff_LD)))")
        #pic = plot(fdlr.τ, delta_new[kF_label,:])
        #pic = plot!(L_fdlr.τ, L_delta_new[kF_label,:])
        #pic = plot(fdlr.n, d_freq[kF_label,:])
        #pic = plot!(L_fdlr.n, L_d_freq[kF_label,:])
        pic = plot(kgrid.grid, delta_0_new)
        pic = plot!(kgrid.grid, L_delta_0_new)
        
        display(pic)
        readline()

        lamu = dot(delta_new, delta)
        #lamu = Normalization(delta_0, delta_0_new, kgrid )
        delta_0_new = delta_0_new+shift*delta_0
        delta_new = delta_new+shift*delta
        #modulus = Normalization(delta_0_new, delta_0_new, kgrid)
        modulus = sqrt(dot(delta_new, delta_new))
        delta_0 = delta_0_new ./ modulus
        delta = delta_new ./ modulus

        L_lamu = dot(L_delta_new, L_delta)
        #lamu = Normalization(delta_0, delta_0_new, kgrid )
        L_delta_0_new = L_delta_0_new+shift*L_delta_0
        L_delta_new = L_delta_new+shift*L_delta
        #modulus = Normalization(delta_0_new, delta_0_new, kgrid)
        modulus = sqrt(dot(L_delta_new, L_delta_new))
        L_delta_0 = L_delta_0_new ./ modulus
        L_delta = L_delta_new ./ modulus
        println("$(lamu),$(L_lamu)")
    end


end




if abspath(PROGRAM_FILE) == @__FILE__
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "a")
    @printf(f, "%.6f\t%.6e\t%d\n", rs, mom_sep, channel)
    close(f)
    if(β<=10000)
    	  fdlr = DLR.DLRGrid(:acorr, 100EF, β, 1e-10)
        #        fdlr = DLR.DLRGrid(:acorr, 1000EF, β, 1e-10)
        L_fdlr = Lehmann.DLR.DLRGrid(:acorr, 10EF, β, 1e-10)
    else
	      fdlr = DLR.DLRGrid(:acorr, 1000EF, β, 1e-10)	
    end
    println(fdlr.τ)
    fdlr2 = DLR.DLRGrid(:acorr, 100EF, β, 1e-10)
    bdlr = DLR.DLRGrid(:corr, 100EF, β, 1e-10)
    L_bdlr = Lehmann.DLR.DLRGrid(:corr, 100EF, β, 1e-10)
    ########## non-uniform kgrid #############
    # Nk = 16
    # order = 8
    # maxK = 10.0 * kF
    # minK = 0.00001 / (β * kF)
    #minK = 0.0000001
    # test1 = zeros(Float64, (2,fdlr.size))
    # for (τi,τ) in enumerate(fdlr.τ)
    #     test1[1,τi] = (β/2.0 - τ)
    #     test1[2,τi] = (β/2.0 - τ)
    # end
    # test_norm = dlr_dot(test1,test1,fdlr)
    # println(test_norm)
    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    # kgrid2 = CompositeGrid(kpanel, order÷2, :cheb)
    # qgrids2 = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order÷2, :gaussian) for k in kgrid.grid] # qgrid for each k
    const kF_label = searchsortedfirst(kgrid.grid, kF)
    const qF_label = searchsortedfirst(qgrids[kF_label].grid,kF)
    println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
    println("$(qF_label), $(qgrids[kF_label].grid[qF_label])")
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    #kernal = dH1_freq(kgrid, qgrids, bdlr, fdlr)
    #kernal = dH1_tau(kgrid, qgrids, fdlr)
    kernal_bare, kernal_freq = legendre_dc(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    L_kernal_bare, L_kernal_freq = legendre_dc(L_bdlr, kgrid, qgrids, kpanel_bose, order_int)
    #kernal_bare, kernal_freq = dH1_freq(kgrid, qgrids, bdlr, fdlr)
    kernal = real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, fdlr.τ, axis=3))
    #L_kernal = real(Lehmann.DLR.matfreq2tau(:corr, L_kernal_freq, L_bdlr, L_fdlr.τ, axis=3))
    L_kernal = real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, L_fdlr.τ, axis=3))    
    # kernal_test =  real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, bdlr.τ, axis=3))
    # kernal_freq_test=  real(DLR.tau2matfreq(:corr, kernal_test, bdlr, bdlr.n, axis=3))
    # println(abs.(kernal_freq_test-kernal_freq)[kF_label,qF_label,:])
    # pic = plot(bdlr.n, kernal_freq[kF_label,qF_label,:]) 
    # pic = plot!(bdlr.n, kernal_freq_test[kF_label,qF_label,:])
    
    # display(pic)
    # readline()
    # println(maximum(abs.(kernal_freq_test-kernal_freq)))
    # println("diff  $(maximum(abs.(kernal-L_kernal))) ")
    
    # println(typeof(kernal))
    #err test section
    kpanel2 =  KPanel(Nk, kF, maxK, minK)
    kgrid_double = CompositeGrid(kpanel2, 2*order, :cheb)
    qgrids_double = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), 2*order, :gaussian) for k in kgrid_double.grid]
    
    #fdlr2 = DLR.DLRGrid(:acorr, 100EF, β, 1e-10) 

    #kernal_bare2, kernal_double = legendre_dc(bdlr, kgrid_double, qgrids_double, kpanel_bose, order_int)
    #kernal2 = real(DLR.matfreq2tau(:corr, kernal_double, bdlr, fdlr.τ, axis=3))

    #initialize delta
    delta = zeros(Float64, (length(kgrid.grid), fdlr.size))
    # for (ki, k) in enumerate(kgrid.grid)
    #     ω = k^2 / (2me) - EF
    #     for (ni, n) in enumerate(fdlr.n)
    #         np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
    #         nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
    #         # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    #         delta[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    #     end
    # end
    # delta =  DLR.matfreq2tau(:acorr, delta, fdlr, fdlr.τ, axis=2)
    delta_0 = zeros(Float64, length(kgrid.grid)) .+ 1.0 

    delta_0, delta, F_throw = Explicit_Solver(kernal, kernal_bare, kgrid, qgrids, fdlr, bdlr)
        


    L_delta = zeros(Float64, (length(kgrid.grid), L_fdlr.size))
    # for (ki, k) in enumerate(kgrid.grid)
    #     ω = k^2 / (2me) - EF
    #     for (ni, n) in enumerate(L_fdlr.n)
    #         np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
    #         nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
    #         # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    #         L_delta[ki, ni] =  Lehmann.Spectral.kernelFermiΩ(nn, ω, β) * Lehmann.Spectral.kernelFermiΩ(np, ω, β)
    #     end
    # end
    # L_delta = Lehmann.DLR.matfreq2tau(:acorr, L_delta, L_fdlr, L_fdlr.τ, axis=2)
    L_delta_0 = zeros(Float64, length(kgrid.grid)) .+1.0
    L_delta =  DLR.tau2dlr(:acorr, delta, fdlr, axis=2)
    L_delta = DLR.dlr2tau(:acorr, L_delta, fdlr, L_fdlr.τ, axis=2)
    coeff_ld = Lehmann.DLR.tau2dlr(:acorr, L_delta, L_fdlr, axis=2)
    println("max: $(maximum(abs.(coeff_ld)))")    
    L_d_test =  real(Lehmann.DLR.tau2matfreq(:acorr, L_delta, L_fdlr, L_fdlr.n, axis=2))
    L_d_test1 = real(Lehmann.DLR.matfreq2tau(:acorr, L_d_test, L_fdlr,L_fdlr.τ, axis=2))
    L_d_test2 = real(Lehmann.DLR.matfreq2tau(:acorr, L_d_test, L_fdlr,fdlr.τ, axis=2))
    println("L fourier  $(maximum(abs.(L_delta-L_d_test1)))")
    println("L fourier  $(maximum(abs.(delta-L_d_test2)))")    
    L_delta_0 = delta_0
    test_Lehmann(delta_0,delta, L_delta_0, L_delta)


    #Δ0_final_low,Δ0_final_high, Δ_final_low, Δ_final_high,  F, lamu = Implicit_Renorm(delta, delta_0, kernal, kernal_bare, kgrid, qgrids, fdlr)
    #Δ0_final = Δ0_final_low .+ Δ0_final_high
    #Δ_final = Δ_final_low .+ Δ_final_high
    #Δ0_double = interpolate(Δ0_final, kgrid, kgrid_double.grid)
    #Δ_double = zeros(Float64, (length(kgrid_double.grid), fdlr.size))
    #for τi in 1:fdlr.size
    #    Δ_double[:, τi] = interpolate(Δ_final[:, τi], kgrid, kgrid_double.grid)
    #end
    #Δ0_final2_low, Δ0_final2_high, Δ_final2_low,Δ_final2_high, F2, lamu2 = Implicit_Renorm( Δ_double, Δ0_double, kernal2, kernal_bare2, kgrid_double, qgrids_double, fdlr)
    #println("$(lamu2-lamu)")
    #Δ0_final, Δ_final, F = Explicit_Solver_err(kernal, kernal2, kernal_bare, kgrid, qgrids, fdlr,fdlr2, bdlr)
    #Δ0_final, Δ_final, F = Explicit_Solver(kernal, kernal_bare, kgrid, qgrids, fdlr, bdlr)
    
    #Δ0_final, Δ_final = Explicit_Solver_inherit( kgrid, qgrids, fdlr, fdlr2, bdlr)
    Δ_freq_low = DLR.tau2matfreq(:acorr, Δ_final_low, fdlr, fdlr.n, axis=2)
    Δ_freq_high = DLR.tau2matfreq(:acorr, Δ_final_high, fdlr, fdlr.n, axis=2)
    F_τ = DLR.tau2dlr(:acorr, F, fdlr, axis=2)
    F_τ = real.(DLR.dlr2tau(:acorr, F_τ, fdlr, extT_grid.grid , axis=2))
    #F_τ = real(DLR.matfreq2tau(:acorr, F_freq, fdlr, extT_grid.grid, axis=2))
    println("F_τ_max:",maximum(F_τ))
    F_ext = zeros(Float64, (length(extK_grid.grid), length(extT_grid.grid)))


   
    #Δ0_final2, Δ_final2, F2 = Implicit_Renorm(kernal, kernal_bare,  kgrid, qgrids, fdlr2)
    #Δ0_final2, Δ_final2, F2 = Explicit_Solver(kernal, kernal_bare, kgrid, qgrids, fdlr2, bdlr)
    
    #Δ_freq2 = DLR.tau2matfreq(:acorr, Δ_final2, fdlr, fdlr.n, axis=2)


    outFileName = rundir*"/f_$(WID).dat"
    f = open(outFileName, "w")

    kpidx = 1
    head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
    x = @view kgrid.grid[head:tail]
    w = @view kgrid.wgrid[head:tail]
    for (ki, k) in enumerate(extK_grid.grid)
        if k > kgrid.panel[kpidx + 1]
            # if q is larger than the end of the current panel, move k panel to the next panel
            while k > kgrid.panel[kpidx + 1]
                global kpidx += 1
            end
            global head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
            global x = @view kgrid.grid[head:tail]
            global w = @view kgrid.wgrid[head:tail]
            @assert kpidx <= kgrid.Np
        end
        for (τi, τ) in enumerate(extT_grid.grid)
            fx = @view F_τ[head:tail, τi] # all F in the same kpidx-th K panel
            F_ext[ki, τi] = barycheb(order, k, fx, w, x) # the interpolation is independent with the panel length
            #@printf("%32.17g  %32.17g  %32.17g\n",extK_grid[ki] ,extT_grid[τi], F_ext[ki, τi])
            @printf(f, "%32.17g  %32.17g  %32.17g\n",extK_grid[ki] ,extT_grid[τi], F_ext[ki, τi])
        end
    end
    


    outFileName = rundir*"/delta_$(WID).dat"
    f = open(outFileName, "w")
    for (ki, k) in enumerate(kgrid.grid)
        for (ni, n) in enumerate(fdlr.τ)
            @printf(f, "%32.17g  %32.17g  %32.17g %32.17g\n",Δ0_final_low[ki], Δ0_final_high[ki], Δ_final_low[ki,ni], Δ_final_high[ki,ni])
        end
    end



    #println(fdlr.n, fdlr.n[fdlr.size ÷ 2 + 1])
    # open(filename, "w") do io
    #     for r in 1:length(kgrid.grid)
    #         @printf(io, "%32.17g  %32.17g  %32.17g\n",kgrid.grid[r] ,Δ0_final[r] ,real(Δ_freq[r, fdlr.size ÷ 2 + 1]))
    #     end
    # end

    # cut_f = 1
    # for r in 1:length(kgrid.grid)
    #     if kgrid.grid[r] < kF
    #         global cut_f = r
    #     end
    # end
    # filename = "./delta_$(WID)_freq.dat"
    # open(filename, "w") do io
    #     for r in 1:fdlr.size
    #         @printf(io, "%32.17g  %32.17g\n",fdlr.n[r],real(Δ_freq[cut_f,r]))
    #     end
    # end
    # step = fdlr.size÷16
    # p = plot(kgrid.grid, Δ0_final[:])
    # for i in 1:6
    #     global p = plot!(p, kgrid.grid, real(Δ_freq[:, fdlr.size ÷ 2 + 1 + i*step]))
    # end
    # #p = plot!(p, kgrid.grid, F_fine[:, 10])
    # display(p)
    # readline()

    # F = calcF(Δ0, Δ, fdlr, kgrid)
    # F_freq = DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
    # Δ_low, Δ_high = Separation_F(F, kgrid, fdlr)
    # q1=20
    # n1=fdlr.size÷2-5
    # n2=fdlr.size÷2+5
    # println(Δ_low[q1,n1:n2])
    # println(Δ_high[q1,n1:n2])
    # println(Δ_low[q1,n1:n2]+Δ_high[q1,n1:n2])
    # println(F_freq[q1,n1:n2])

    # F_test1 =  DLR.matfreq2tau(:acorr, Δ_low, fdlr, fdlr.τ, axis=2)
    # F_test1 =  DLR.tau2matfreq(:acorr, F_test1, fdlr, fdlr.n, axis=2)
    # println(F_test1[q1,n1:n2])
    # println(maximum(abs.(real(F_test1)-Δ_low)),",", maximum(abs.(imag(F_test1))))
end
