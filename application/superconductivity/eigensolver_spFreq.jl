"""
    Power method, damp interation and implicit renormalization
    """
#module eigensolver
#using QuantumStatistics
using QuantumStatistics: σx, σy, σz, σ0, Grid,FastMath, Utility, TwoPoint #, DLR, Spectral
using Lehmann
using LinearAlgebra
using DelimitedFiles
using Printf
#using Gaston
#using Plots
using Statistics

srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

if !(@isdefined β)
    include(rundir*"/parameter.jl")
    using .parameter
end

include("eigen.jl")
include("grid.jl")

Ω_c = freq_sep

function AcorrSepWeight(fdlr, Ω_c)
    n_c = Base.floor(Int, Ω_c*β/(2π) -0.5)
    ASW = zeros(Float64, fdlr.size)
    for (ξi, ξ) in enumerate(fdlr.ω)
        for m in 1:n_c
            ASW[ξi] += Spectral.kernelAnormalCorrΩ(m, ξ, β)
        end
    end

    return ASW
end

function ConvSepWeight(fdlr, bdlr, Ω_c)
    n_c = Base.floor(Int, Ω_c*β/(2π) -0.5)
    conv_mat = zeros(Float64, (fdlr.size, bdlr.size, fdlr.size))

    for (ni, n) in enumerate(fdlr.n)
        for (ωi, ω) in enumerate(bdlr.ω)
            for (ξi, ξ) in enumerate(fdlr.ω)
                for m in 1:n_c
                    conv_mat[ni, ωi, ξi] += Spectral.kernelAnormalCorrΩ(m, ξ, β)*Spectral.kernelCorrΩ(n-m, ω, β)
                end
            end
        end
    end

    return conv_mat
end

function calcF_freqSep(Δ_freq, Σ, fdlr, k::CompositeGrid)
    F = zeros(ComplexF64, (length(k.grid), fdlr.size))
    for (ki, k) in enumerate(k.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            #F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, Σ[ki,ni], β) * Spectral.kernelFermiΩ(np, ω, Σ[ki,ni], β)
            F[ki, ni] = (Δ_freq[ki, ni]) /(((2*np+1)*π/β-imag(Σ[ki,ni]))^2 + (ω + real(Σ[ki,ni]))^2)
            #F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
           
        end

    end
    
    #F = DLR.matfreq2tau(:acorr, F, fdlr, fdlr.τ, axis=2)
    @assert isfinite(sum(F)) "fail to calculate F"
    return  real(F) # return F in matfreq space
end

function calcΔ_freqSep(F_freq, kernal_freq, kernal_bare, bdlr, fdlr, kgrid, qgrids, ASW, CSW, lamu)
    """
        Calculate new Δ with F. Here
            F=λF_1+F_2,
        so that
            Δ = - TΣΓ(F_1+F_2) = -TΣΓF-(1-λ)/λ*TΣΓP_1F
        TΣΓF is calculated as previous,
        second term is calculated with ASW and CSW

        F and kernal should be in matfreq space(original form) for ASW and CSW calc.
        return result in matfreq space
    """
    F_tau = DLR.matfreq2tau(:acorr, F_freq, fdlr, fdlr.τ, axis=2)

    kernal_tau = real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, fdlr.τ, axis=3))
    kernal_dlr = (DLR.matfreq2dlr(:corr, kernal_freq, bdlr, axis=3))

    Δ0, Δ_tau = calcΔ(F_tau, kernal_tau, kernal_bare, fdlr, kgrid, qgrids)
    Δ_freq = DLR.tau2matfreq(:acorr, Δ_tau, fdlr, fdlr.n, axis=2)

    Δ_freq2 = zeros(Float64, (length(kgrid.grid), fdlr.size))

    for (ki, k) in enumerate(kgrid.grid)
        
        kpidx = 1 # panel index of the kgrid
        head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
        x = @view kgrid.grid[head:tail]
        w = @view kgrid.wgrid[head:tail]

        for (qi, q) in enumerate(qgrids[ki].grid)
            
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

            FF = zeros(Float64, fdlr.size)
            for (mi, m) in enumerate(fdlr.n)
                fx = @view F_freq[head:tail, mi] # all F in the same kpidx-th K panel
                FF[mi] = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length
            end
            FF = DLR.matfreq2dlr(:acorr, FF, fdlr, axis=1)

            sing_result = 0.0
            for (ξi, ξ) in enumerate(fdlr.ω)
                sing_result += FF[ξi] * ASW[ξi]
            end
            sing_result = real(sing_result)
            @assert isfinite(sing_result) "fail to calculate sing_result"
            for (ni, n) in enumerate(fdlr.n)

                conv_result = 0.0+0.0im
                for (ωi, ω) in enumerate(bdlr.ω)
                    for (ξi, ξ) in enumerate(fdlr.ω)
                        conv_result += kernal_dlr[qi, ki, ωi] * FF[ξi] * CSW[ni, ωi, ξi]
                    end
                end
                conv_result = real(conv_result)
                @assert isfinite(conv_result) "fail to calculate conv_result"
                wq = qgrids[ki].wgrid[qi]
                #Δ[ki, τi] += dH1(k, q, τ) * FF * wq
                Δ_freq2[ki, ni] += conv_result * wq / β + sing_result * kernal_bare[ki, qi] * wq / β
                @assert isfinite(Δ_freq2[ki, ni]) "fail to calculate Δ_freq2[ki, ni]=$(Δ_freq2[ki, ni]), ki=$(ki), ni=$(ni),qi=$(qi), conv_result=$(conv_result), sing_result=$(sing_result), bare=$(kernal_bare[ki,qi]), wq=$(wq),β=$(β)"
            end
        end
    end

    @assert isfinite(sum(Δ_freq)) "fail to calculate Δ_freq"
    @assert isfinite(sum(Δ_freq2)) "fail to calculate Δ_freq2"

    return real(Δ_freq .+ (1-lamu)/lamu .* Δ_freq2)

end

"""
    Function Separation

    Separate Gap function to low and high energy parts with given rule
"""

function Freq_Sep(delta, fdlr, i_sep)
    low=zeros(Float64, (length(kgrid.grid), fdlr.size))
    high=zeros(Float64, (length(kgrid.grid), fdlr.size))

    low[:, 1:i_sep] = delta[:, 1:i_sep]
    high[:, i_sep+1:end] = delta[:, i_sep+1:end] 
    return low, high
end

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
        @assert isfinite(sum) "fail normaliztion of Δ0, DD1=$(DD1),DD2=$(DD2),wq=$(wq)"
    end
    return sum
end


function Separation(delta0, delta, k::CompositeGrid,  fdlr)
    low=zeros(Float64, ( length(k.grid), fdlr.size))
    high=zeros(Float64, ( length(k.grid), fdlr.size))
    low0=zeros(Float64,  length(k.grid))
    high0=zeros(Float64,  length(k.grid))
    kindex_left = searchsortedfirst(k.panel, kF-mom_sep)
    kindex_right = searchsortedlast(k.panel, kF+mom_sep)
    #println("separation point:$(k.panel[kindex_left]),$(k.panel[kindex_right])")
    head = idx(kindex_left, 1, order)
    tail = idx(kindex_right, 1, order)-1

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

    
    return delta_0_low, delta_0_high, delta_low, delta_high
end

function Separation_F(F_in,  k::CompositeGrid,  fdlr)
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


function Implicit_Renorm(Δ, Δ_0 ,kernal, kernal_bare, Σ, kgrid, qgrids, fdlr )
    
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
        F=calcF(delta_0, delta, Σ, fdlr, kgrid)
        
        n=n+1
        delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)


	      delta_0_low_new, delta_0_high_new, delta_low_new, delta_high_new = Separation(delta_0_new, delta_new, kgrid, fdlr)
        
	      if(Looptype==0)
            accm=accm+1
            d_0_accm = d_0_accm + delta_0_high_new 
            d_accm = d_accm + delta_high_new
            delta_0_high = d_0_accm ./ accm
            delta_high = d_accm ./ accm
        else
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

    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "a")
    @printf(f, "%32.17g\n", lamu)
    close(f)
    #F=calcF(delta_0, delta, fdlr, kgrid)
    #delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)


    return delta_0_low, delta_0_high, delta_low, delta_high, F, lamu
end

function Implicit_Renorm_Freq(Δ, kernal_freq, kernal_bare, Σ, kgrid, qgrids, fdlr, bdlr)
    
    NN=100000
    rtol=1e-6
    Looptype=1
    n=0
    err=1.0 
    accm=0
    shift=5.0
    lamu0=-2.0
    lamu=0.5
    n_change=10  #steps of power method iteration in one complete loop
    n_change2=10+10 #total steps of one complete loop
    delta = Δ

    ASW = AcorrSepWeight(fdlr, Ω_c)
    CSW = ConvSepWeight(fdlr,bdlr, Ω_c)
    println(size(ASW), size(CSW))

    #Separate Delta
    i_sep = searchsortedfirst(fdlr.ωn, Ω_c)
    d_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    delta_low, delta_high = Freq_Sep(delta, fdlr, i_sep)
    F = zeros(Float64, (length(kgrid.grid), fdlr.size))
    while(n<NN && err>rtol)
        F=calcF_freqSep(delta, Σ, fdlr, kgrid)
        
        n=n+1
        delta_new =  calcΔ_freqSep(F, kernal_freq, kernal_bare,bdlr, fdlr , kgrid, qgrids, ASW,CSW, lamu)./(-4*π*π)


	      delta_low_new, delta_high_new = Freq_Sep(delta_new, fdlr, i_sep)
        
	      if(Looptype==0)
            accm=accm+1
            d_accm = d_accm + delta_high_new
            delta_high = d_accm ./ accm
        else
            lamu = Normalization(delta_low[:, 1], delta_low_new[:, 1], kgrid, qgrids )
            delta_low_new = delta_low_new+shift*delta_low
            modulus = Normalization(delta_low_new[:, 1], delta_low_new[:, 1], kgrid, qgrids )
            @assert modulus>0
            modulus = sqrt(modulus)
            delta_low = delta_low_new ./ modulus
            #println(lamu)
        end
        delta = lamu .* delta_low .+ delta_high
        if(n%n_change2==n_change)
            Looptype=(Looptype+1)%2
        elseif(n%n_change2==0)
            accm = 0
            d_accm = d_accm .* 0
            err=abs(lamu-lamu0)
            lamu0=lamu
            println(lamu)
            Looptype=(Looptype+1)%2
        end
        
    end

    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "a")
    @printf(f, "%32.17g\n", lamu)
    close(f)
    #F=calcF(delta_0, delta, fdlr, kgrid)
    #delta_0_new, delta_new =  calcΔ(F, kernal, kernal_bare, fdlr , kgrid, qgrids)./(-4*π*π)


    return delta_low, delta_high, F, lamu
end





if abspath(PROGRAM_FILE) == @__FILE__
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    outFileName = rundir*"/flow_$(WID).dat"
    if !(isfile(outFileName))
        f = open(outFileName, "a")
        @printf(f, "%.6e\t%.6f\t%.6e\t%d\n", β, rs, mom_sep, channel)
        close(f)
    end    
    fdlr = DLR.DLRGrid(:acorr, fEUV, β, 1e-8)
    println(fdlr.τ)
    fdlr2 = DLR.DLRGrid(:acorr, 100EF, β, 1e-8)
    bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10)

    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    const kF_label = searchsortedfirst(kgrid.grid, kF)
    const qF_label = searchsortedfirst(qgrids[kF_label].grid,kF)
    
    println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
    println("$(qF_label), $(qgrids[kF_label].grid[qF_label])")
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    kernal_bare, kernal_freq = legendre_dc(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    println(kernal_freq[kF_label,qF_label,:])

    kernal = real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, fdlr.τ, axis=3))
    println(size(kernal_freq),size(kernal))

    #err test section
    kpanel2 =  KPanel(Nk, kF, maxK, minK)
    kgrid_double = CompositeGrid(kpanel2, 2*order, :cheb)
    qgrids_double = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), 2*order, :gaussian) for k in kgrid_double.grid]
    
    #initialize delta
    delta = zeros(Float64, (length(kgrid.grid), fdlr.size)) .+ 1.0
    delta_0 = zeros(Float64, length(kgrid.grid)) .+ 1.0 


    if(sigma_type == :none)
        Σ = (0.0+0.0im) * delta
        Δ_final_low, Δ_final_high,  F, lamu = Implicit_Renorm_Freq(delta, kernal_freq, kernal_bare, Σ, kgrid, qgrids, fdlr, bdlr)
    else
        w0_label = 0
        dataFileName = rundir*"/sigma_$(WID).dat"
        f = open(dataFileName, "r")
        Σ_raw = readdlm(f)
        Σ  = transpose(reshape(Σ_raw[:,1],(fdlr.size,length(kgrid.grid)))) + transpose(reshape(Σ_raw[:,2],(fdlr.size,length(kgrid.grid))))*im
        Δ_final_low, Δ_final_high,  F, lamu = Implicit_Renorm_Freq(delta, kernal_freq, kernal_bare, Σ, kgrid, qgrids, fdlr, bdlr)
    end

    if method_type == :implicit
        Δ_out = lamu .* Δ_final_low .+ Δ_final_high
        F_τ = DLR.matfreq2dlr(:acorr, F, fdlr, axis=2)
        F_τ = real.(DLR.dlr2tau(:acorr, F_τ, fdlr, extT_grid.grid , axis=2))
        #F_τ = real(DLR.matfreq2tau(:acorr, F_freq, fdlr, extT_grid.grid, axis=2))
        println("F_τ_max:",maximum(F_τ))
        F_ext = zeros(Float64, (length(extK_grid.grid), length(extT_grid.grid)))

        
        outFileName = rundir*"/delta_$(WID).dat"
        f = open(outFileName, "w")
        for (ki, k) in enumerate(kgrid.grid)
            for (ni, n) in enumerate(fdlr.τ)
                @printf(f, "%32.17g %32.17g\n",Δ_final_low[ki,ni], Δ_final_high[ki,ni])
            end
        end
        #println(Δ_out)
        outFileName = rundir*"/flow_$(WID).dat"
        f = open(outFileName, "a")
        @printf(f, "%32.17g\t%32.17g\n",Δ_out[1], Δ_out[end] )
        close(f)
    end

end
