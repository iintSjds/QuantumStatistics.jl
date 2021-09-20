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
include("DLR_FreqConv.jl")
using .FreqConv

Ω_c = freq_sep

function calcF_freqSep(Δ_freq, Σ, fdlr, k::CompositeGrid)
    F = zeros(Float64, (length(k.grid), fdlr.size))
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
    return  (F) # return F in matfreq space
end

function kernel_sep(kernel, cm)
    fdlr, bdlr = cm.fdlr, cm.bdlr
    low_mat, high_mat = cm.sep_mat, cm.high_mat

    kernel_dlr = DLR.matfreq2dlr(:corr, kernel, bdlr, axis = 3)
    # seems a bug of DLR: exchange axis 1 and 2 after mat2dlr or dlr2mat. mat2tau and reverse is ok.

    kernel_low = zeros(Float64, (size(kernel)[1], size(kernel)[2], fdlr.size, fdlr.size))
    kernel_high = zeros(Float64, (size(kernel)[1], size(kernel)[2], fdlr.size, fdlr.size))

    for ki in 1:size(kernel)[1]
        for qi in 1:size(kernel)[2]
            for ni in 1:fdlr.size
                for ξi in 1:fdlr.size
                    for mi in 1:bdlr.size
                        kernel_low[ki, qi, ni, ξi] = kernel_dlr[qi, ki, mi]*low_mat[ni, mi, ξi]
                        kernel_high[ki, qi, ni, ξi] = kernel_dlr[qi, ki, mi]*high_mat[ni, mi, ξi]
                    end
                end
            end
        end
    end

    return kernel_low, kernel_high
end

function calcΔ_freqSep(F_low, F_high, kernel_low, kernel_high, kernel_bare, kgrid, qgrids, cm)
    """
        Calculate new Δ with F.
        Δ = (KP_1)F_1 + (KP_2)F_2

        F and kernel should be in matfreq space(original form) for ASW and CSW calc.
        return result in matfreq space
    """
    fdlr, bdlr = cm.fdlr, cm.bdlr
    #F_tau = DLR.matfreq2tau(:acorr, F_freq, fdlr, fdlr.τ, axis=2)
    Fl_dlr = DLR.matfreq2dlr(:acorr, F_low, fdlr, axis=2)
    Fh_dlr = DLR.matfreq2dlr(:acorr, F_high, fdlr, axis=2)

    Δ_freq = zeros(Float64, (length(kgrid.grid), fdlr.size))

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

            FFl = zeros(Float64, fdlr.size)
            FFh = zeros(Float64, fdlr.size)
            for (mi, m) in enumerate(fdlr.n)
                fx = @view F_low[head:tail, mi] # all F in the same kpidx-th K panel
                FFl[mi] = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length
                fx = @view F_high[head:tail, mi] # all F in the same kpidx-th K panel
                FFh[mi] = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length
            end
            FFl = DLR.matfreq2dlr(:acorr, FFl, fdlr, axis=1)
            FFh = DLR.matfreq2dlr(:acorr, FFh, fdlr, axis=1)

            sing_result = 0.0
            for (ξi, ξ) in enumerate(fdlr.ω)
                sing_result += FFl[ξi] * cm.asw_low[ξi] + FFh[ξi] * cm.asw_high[ξi]
            end
            sing_result = real(sing_result)
            @assert isfinite(sing_result) "fail to calculate sing_result"

            for (ni, n) in enumerate(fdlr.n)

                conv_result = 0.0
                for (ξi, ξ) in enumerate(fdlr.ω)
                    conv_result += kernel_low[ki, qi, ni, ξi] * FFl[ξi] + kernel_high[ki, qi, ni, ξi] * FFh[ξi]
                end
                @assert isfinite(conv_result) "fail to calculate conv_result"
                wq = qgrids[ki].wgrid[qi]
                #Δ[ki, τi] += dH1(k, q, τ) * FF * wq
                Δ_freq[ki, ni] += conv_result * wq + sing_result * kernel_bare[ki, qi] * wq
                @assert isfinite(Δ_freq[ki, ni]) "fail to calculate Δ_freq[ki, ni]=$(Δ_freq[ki, ni]), ki=$(ki), ni=$(ni),qi=$(qi), conv_result=$(conv_result), sing_result=$(sing_result), bare=$(kernel_bare[ki,qi]), wq=$(wq),β=$(β)"
            end
        end
    end

    @assert isfinite(sum(Δ_freq)) "fail to calculate Δ_freq"

    return Δ_freq

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


function Implicit_Renorm_Freq(Δ, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
    
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

    n_c = Base.floor(Int,Ω_c/(2π)*β)
    cm = FreqConv.ConvMat(bdlr, fdlr, n_c)

    kernel_low, kernel_high = kernel_sep(kernel_freq, cm)

    #Separate Delta
    i_sep = searchsortedfirst(fdlr.ωn, Ω_c)
    d_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    delta_low, delta_high = delta, delta
    F_low = zeros(Float64, (length(kgrid.grid), fdlr.size))
    F_high = zeros(Float64, (length(kgrid.grid), fdlr.size))

    while(n<NN && err>rtol)
        F_low=calcF_freqSep(delta_low, Σ, fdlr, kgrid)
        F_high=calcF_freqSep(delta_high, Σ, fdlr, kgrid)
        
        n=n+1
        delta_new =  calcΔ_freqSep(F_low, F_high, kernel_low, kernel_high, kernel_bare, kgrid, qgrids, cm)./(-4*π*π)

	      if(Looptype==0)
            accm=accm+1
            d_accm = d_accm + delta_new
            delta_high = d_accm ./ accm
        else
            lamu = Normalization(delta_low[:, 1], delta_new[:, 1], kgrid, qgrids )
            delta_new = delta_new+shift*delta_low
            modulus = Normalization(delta_new[:, 1], delta_new[:, 1], kgrid, qgrids )
            @assert modulus>0
            modulus = sqrt(modulus)
            delta_low = delta_new ./ modulus
            #println(lamu)
        end
        delta = lamu .* Freq_Sep(delta_low, fdlr, i_sep)[1] .+ Freq_Sep(delta_high, fdlr, i_sep)[2]
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
    #delta_0_new, delta_new =  calcΔ(F, kernel, kernel_bare, fdlr , kgrid, qgrids)./(-4*π*π)


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

    kernel_bare, kernel_freq = legendre_dc(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    println(kernel_freq[kF_label,qF_label,:])

    kernel = real(DLR.matfreq2tau(:corr, kernel_freq, bdlr, fdlr.τ, axis=3))
    println(size(kernel_freq),size(kernel))

    #err test section
    kpanel2 =  KPanel(Nk, kF, maxK, minK)
    kgrid_double = CompositeGrid(kpanel2, 2*order, :cheb)
    qgrids_double = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), 2*order, :gaussian) for k in kgrid_double.grid]
    
    #initialize delta
    delta = zeros(Float64, (length(kgrid.grid), fdlr.size)) .+ 1.0
    delta_0 = zeros(Float64, length(kgrid.grid)) .+ 1.0 


    if(sigma_type == :none)
        Σ = (0.0+0.0im) * delta
        Δ_final_low, Δ_final_high,  F, lamu = Implicit_Renorm_Freq(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
    else
        w0_label = 0
        dataFileName = rundir*"/sigma_$(WID).dat"
        f = open(dataFileName, "r")
        Σ_raw = readdlm(f)
        Σ  = transpose(reshape(Σ_raw[:,1],(fdlr.size,length(kgrid.grid)))) + transpose(reshape(Σ_raw[:,2],(fdlr.size,length(kgrid.grid))))*im
        Δ_final_low, Δ_final_high,  F, lamu = Implicit_Renorm_Freq(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
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