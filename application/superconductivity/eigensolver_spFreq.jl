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
    println("kernel sep start:sep")
    fdlr, bdlr = cm.fdlr, cm.bdlr
    low_mat, high_mat = cm.sep_mat, cm.high_mat

    #kernel_dlr = DLR.matfreq2dlr(:corr, kernel, bdlr, axis = 3)
    kernel_dlr = similar(kernel)
    for ki in 1:size(kernel)[1]
        for qi in 1:size(kernel)[2]
            kernel_dlr[ki, qi, :] = DLR.matfreq2dlr(:corr, kernel[ki, qi, :], bdlr, axis = 1)
        end
    end

    # seems a bug of DLR: exchange axis 1 and 2 after mat2dlr or dlr2mat. mat2tau and reverse is ok.

    # kernel_low = zeros(Float64, (size(kernel)[1], size(kernel)[2], fdlr.size, fdlr.size))
    # kernel_high = zeros(Float64, (size(kernel)[1], size(kernel)[2], fdlr.size, fdlr.size))
    kernel_low = zeros(Float64, (fdlr.size, fdlr.size, size(kernel)[2], size(kernel)[1]))
    kernel_high = zeros(Float64, (fdlr.size, fdlr.size, size(kernel)[2], size(kernel)[1]))

    for ki in 1:size(kernel)[1]
        for qi in 1:size(kernel)[2]
            for ni in 1:fdlr.size
                for ξi in 1:fdlr.size
                    for mi in 1:bdlr.size
                        kernel_low[ξi, ni, qi, ki] += kernel_dlr[ki, qi, mi]*low_mat[ni, mi, ξi]
                        kernel_high[ξi, ni, qi, ki] += kernel_dlr[ki, qi, mi]*high_mat[ni, mi, ξi]
                    end
                end
            end
        end
        println("$(ki)/$(size(kernel)[1])")
    end

    println("kernel sep end")
    return kernel_low, kernel_high
end

function kernel_sep_full(kernel, cm)
    println("kernel sep start:full")
    fdlr, bdlr = cm.fdlr, cm.bdlr

    #kernel_dlr = DLR.matfreq2dlr(:corr, kernel, bdlr, axis = 3)
    kernel_dlr = similar(kernel)
    for ki in 1:size(kernel)[1]
        for qi in 1:size(kernel)[2]
            kernel_dlr[ki, qi, :] = DLR.matfreq2dlr(:corr, kernel[ki, qi, :], bdlr, axis = 1)
        end
    end

    println("max dlr coef:$(maximum(abs.(real(kernel_dlr))))")

    # seems a bug of DLR: exchange axis 1 and 2 after mat2dlr or dlr2mat. mat2tau and reverse is ok.

    # kernel_full = zeros(Float64, (size(kernel)[1], size(kernel)[2], fdlr.size, fdlr.size))
    kernel_full = zeros(Float64, (fdlr.size, fdlr.size, size(kernel)[2], size(kernel)[1]))

    for ki in 1:size(kernel)[1]
        for qi in 1:size(kernel)[2]
            for ni in 1:fdlr.size
                for ξi in 1:fdlr.size
                    for mi in 1:bdlr.size
                        kernel_full[ξi, ni, qi, ki] += kernel_dlr[ki, qi, mi]*cm.full_mat[ni, mi, ξi]
                    end
                end
            end
        end
        println("$(ki)/$(size(kernel)[1])")
    end

    println("$(kernel_full[1,1,:,1])")
    println("kernel sep end")
    return kernel_full
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
    # println("max F:$(maximum(abs.(real(F_low)))), $(maximum(abs.(real(F_high))))")
    # println("max dlr coef:$(maximum(abs.(real(Fl_dlr)))), $(maximum(abs.(real(Fh_dlr))))")
    # println("max dlr imag:$(maximum(abs.(imag(Fl_dlr)))), $(maximum(abs.(imag(Fh_dlr))))")

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
                fxl = @view F_low[head:tail, mi] # all F in the same kpidx-th K panel
                FFl[mi] = barycheb(order, q, fxl, w, x) # the interpolation is independent with the panel length
                fxh = @view F_high[head:tail, mi] # all F in the same kpidx-th K panel
                FFh[mi] = barycheb(order, q, fxh, w, x) # the interpolation is independent with the panel length
            end
            FFl = real(DLR.matfreq2dlr(:acorr, FFl, fdlr, axis=1))
            FFh = real(DLR.matfreq2dlr(:acorr, FFh, fdlr, axis=1))

            sing_result = 0.0
            for (ξi, ξ) in enumerate(fdlr.ω)
                sing_result += FFl[ξi] * cm.asw_low[ξi] + FFh[ξi] * cm.asw_high[ξi]
            end
            sing_result = real(sing_result)
            @assert isfinite(sing_result) "fail to calculate sing_result"

            for (ni, n) in enumerate(fdlr.n)

                conv_result = 0.0
                for (ξi, ξ) in enumerate(fdlr.ω)
                    conv_result += kernel_low[ξi, ni, qi, ki] * FFl[ξi] + kernel_high[ξi, ni, qi, ki] * FFh[ξi]
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

function calcΔ_freqFull(F, kernel, kernel_bare, kgrid, qgrids, cm)
    """
        Calculate new Δ with F.
        Δ = (KP_1)F_1 + (KP_2)F_2

        F and kernel should be in matfreq space(original form) for ASW and CSW calc.
        return result in matfreq space
    """
    fdlr, bdlr = cm.fdlr, cm.bdlr
    #F_tau = DLR.matfreq2tau(:acorr, F_freq, fdlr, fdlr.τ, axis=2)
    F_dlr = DLR.matfreq2dlr(:acorr, F, fdlr, axis=2)
    Ft = DLR.matfreq2tau(:acorr, F, fdlr, fdlr.τ, axis=2)

    # println("max dlr coef:$(maximum(abs.(real(F_dlr))))")

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

            FF = zeros(Float64, fdlr.size)
            # for (mi, m) in enumerate(fdlr.n)
            #     fx = @view F[head:tail, mi] # all F in the same kpidx-th K panel
            #     FF[mi] = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length
            # end
            # FF = real(DLR.matfreq2dlr(:acorr, FF, fdlr, axis=1))
            for (ti, t) in enumerate(fdlr.τ)
                fx = @view Ft[head:tail, ti] # all F in the same kpidx-th K panel
                FF[ti] = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length
            end
            FF = real(DLR.tau2dlr(:acorr, FF, fdlr, axis=1))

            sing_result = 0.0
            for (ξi, ξ) in enumerate(fdlr.ω)
                sing_result += FF[ξi] * cm.asw_full[ξi]
            end
            sing_result = real(sing_result)
            @assert isfinite(sing_result) "fail to calculate sing_result"

            for (ni, n) in enumerate(fdlr.n)

                conv_result = 0.0
                for (ξi, ξ) in enumerate(fdlr.ω)
                    conv_result += kernel[ξi, ni, qi, ki] * FF[ξi]
                    # for (ωi, ω) in enumerate(bdlr.ω)
                    #     conv_result += kernel[qi, ki, ωi] * cm.full_mat[ni, ωi, ξi] * FF[ξi]
                    # end
                end
                # if ki == 56 && qi == 58
                #     println("conv_result:$(conv_result)")
                #     println("sing_result:$(sing_result)")
                #     #println("kernel_bare:$(kernel_bare[ki,qi])")
                # end
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
    Looptype=0
    n=0
    err=1.0 
    accm=0.0
    shift=5.0
    lamu0=-2.0
    lamu=0.5
    n_change=3  #steps of power method iteration in one complete loop
    n_change2=3+3 #total steps of one complete loop
    delta = Δ

    kF_label = searchsortedfirst(kgrid.grid, kF)

    n_c = Base.floor(Int,Ω_c/(2π)*β)
    println("Sep:$(Ω_c), n_c:$(n_c)")
    cm = FreqConv.ConvMat(bdlr, fdlr, n_c)

    kernel_low, kernel_high = kernel_sep(kernel_freq, cm)

    #Separate Delta
    i_sep = searchsortedfirst(fdlr.ωn, Ω_c)
    d_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    delta_low, delta_high = delta .* 1.0, delta .* 1.0
    F_low = zeros(Float64, (length(kgrid.grid), fdlr.size))
    F_high = zeros(Float64, (length(kgrid.grid), fdlr.size))

    while(n<NN && err>rtol)
        #println(delta_low[kF_label,:])
        #println(delta_high[kF_label,:])
        F_low=calcF_freqSep(delta_low, Σ, fdlr, kgrid)
        F_high=calcF_freqSep(delta_high, Σ, fdlr, kgrid)
        
        n=n+1
        delta_new =  calcΔ_freqSep(F_low, F_high, kernel_low, kernel_high, kernel_bare, kgrid, qgrids, cm)./(-4*π*π)

	      # if(Looptype==0)
        #     accm=accm+1
        #     d_accm = d_accm + delta_new
        #     delta_high = d_accm ./ accm
        # else
        #     lamu = Normalization(delta_low[:, 1], delta_new[:, 1], kgrid, qgrids )
        #     delta_new = delta_new+shift*delta_low
        #     modulus = Normalization(delta_new[:, 1], delta_new[:, 1], kgrid, qgrids )
        #     @assert modulus>0
        #     modulus = sqrt(modulus)
        #     delta_low = delta_new ./ modulus
        #     println(lamu)
        # end
        # delta = lamu .* Freq_Sep(delta_low, fdlr, i_sep)[1] .+ Freq_Sep(delta_high, fdlr, i_sep)[2]
        # if(n%n_change2==n_change)
        #     Looptype=(Looptype+1)%2
        # elseif(n%n_change2==0)
        #     accm = 0
        #     d_accm = d_accm .* 0
        #     err=abs(lamu-lamu0)
        #     lamu0=lamu
        #     #println(lamu)
        #     Looptype=(Looptype+1)%2
        # end

        accm=accm * 0.8 + 1.0
        d_accm = d_accm .* 0.8 .+ delta_new
        delta_high = d_accm ./ accm

        lamu = Normalization(delta_low[:, 1], delta_new[:, 1], kgrid, qgrids )
        delta_new = delta_new+shift*delta_low
        modulus = Normalization(delta_new[:, 1], delta_new[:, 1], kgrid, qgrids )
        @assert modulus>0
        modulus = sqrt(modulus)
        delta_low = delta_new ./ modulus
        println(lamu)
        delta = lamu .* Freq_Sep(delta_low, fdlr, i_sep)[1] .+ Freq_Sep(delta_high, fdlr, i_sep)[2]
        err=abs(lamu-lamu0)/(abs(lamu)+0.001)
        lamu0=lamu

	if (n%20)==10
	   outFileName = rundir*"/lamu_$(WID).dat"
   	   f = open(outFileName, "a")
	   @printf(f, "%32.17g\n", lamu)
	   close(f)	    
	end

    end

    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "a")
    @printf(f, "%32.17g\n", lamu)
    close(f)
    #F=calcF(delta_0, delta, fdlr, kgrid)
    #delta_0_new, delta_new =  calcΔ(F, kernel, kernel_bare, fdlr , kgrid, qgrids)./(-4*π*π)
    F = lamu .* Freq_Sep(F_low, fdlr, i_sep)[1] .+ Freq_Sep(F_high, fdlr, i_sep)[2]


    return delta_low, delta_high, F, lamu
end

function Explicit_Freq(Δ, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
    
    NN=100000
    #NN=20
    rtol=1e-6
    Looptype=1
    n=0
    err=1.0 
    shift=5.0
    lamu0=-2.0
    lamu=0.5
    delta = Δ

    kF_label = searchsortedfirst(kgrid.grid, kF)

    n_c = Base.floor(Int,Ω_c/(2π)*β)
    println("Sep:$(Ω_c), n_c:$(n_c)")
    cm = FreqConv.ConvMat(bdlr, fdlr, n_c)

    kernel = kernel_sep_full(kernel_freq, cm)


    #Separate Delta
    F = zeros(Float64, (length(kgrid.grid), fdlr.size))

    while(n<NN && err>rtol)
        F=calcF_freqSep(delta, Σ, fdlr, kgrid)

        
        n=n+1
        delta_new =  calcΔ_freqFull(F, kernel, kernel_bare, kgrid, qgrids, cm)./(-4*π*π)
        # println(delta_new[kF_label,:])

        lamu = Normalization(delta[:, 1], delta_new[:, 1], kgrid, qgrids )
        delta_new = delta_new+shift*delta
        modulus = Normalization(delta_new[:, 1], delta_new[:, 1], kgrid, qgrids )
        @assert modulus>0
        modulus = sqrt(modulus)
        delta = delta_new ./ modulus
        println(lamu)
        
    end

    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "a")
    @printf(f, "%32.17g\n", lamu)
    close(f)
    #F=calcF(delta_0, delta, fdlr, kgrid)
    #delta_0_new, delta_new =  calcΔ(F, kernel, kernel_bare, fdlr , kgrid, qgrids)./(-4*π*π)


    return delta, F, lamu
end


function delta_init(fdlr, kgrid)
    delta = zeros(Float64, (length(kgrid.grid), fdlr.size)) .+ 1.0
    for (ki, k) in enumerate(kgrid.grid)
        for (ni, n) in enumerate(fdlr.n)
            ω = π*(2*n+1)/β
            ξ = k^2 - EF
            delta[ki, ni] = (1.0 - 2.0*ω^2/(ω^2+Ω_c^2)) / (Ω_c^2+ξ^2)
        end
    end
    return delta
end

function test_calcΔ(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
    kF_label = searchsortedfirst(kgrid.grid, kF)
    qF_label = searchsortedfirst(qgrids[kF_label].grid, kF)

    n_c = Base.floor(Int,Ω_c/(2π)*β)
    println("Sep:$(Ω_c), n_c:$(n_c)")
    cm = FreqConv.ConvMat(bdlr, fdlr, n_c)


    kernel_t = real(DLR.matfreq2tau(:corr, kernel_freq, bdlr, fdlr.τ, axis=3))
    kernel_dlr = (DLR.matfreq2dlr(:corr, kernel_freq, bdlr, axis=3))
    println("compare kernel dlr")
    println(real(kernel_dlr[qF_label,kF_label,:]))
    println(real(DLR.matfreq2dlr(:corr, kernel_freq[kF_label,qF_label,:], bdlr, axis=1)))
    #kernel = zeros(Float64, (size(kernel_freq)[1], size(kernel_freq)[2], fdlr.size, fdlr.size))
    #kernel_t = 0.0 .* kernel_t

    F = calcF_freqSep(delta, Σ, fdlr, kgrid)
    Ft = DLR.matfreq2tau(:acorr, F, fdlr, fdlr.τ, axis=2)
    F_dlr = DLR.matfreq2dlr(:acorr, F, fdlr, axis=2)

    kernel = kernel_sep_full(kernel_freq, cm)
    kernel_low, kernel_high = kernel_sep(kernel_freq, cm)
    #kernel = DLR.matfreq2dlr(:corr, kernel_freq, bdlr, axis=3)

    println(FreqConv.freq_conv(kernel_freq[kF_label,qF_label,:], F[kF_label,:], cm, :full))
    println(real(DLR.tau2matfreq(:acorr, kernel_t[kF_label,qF_label,:] .* Ft[kF_label,:], fdlr, fdlr.n)))
    println(kernel[kF_label,qF_label,:,:]*real(F_dlr[kF_label,:]))

    delta_new3 = calcΔ_freqSep(F, F, kernel_low, kernel_high, kernel_bare, kgrid, qgrids, cm)
    delta_new1 = calcΔ_freqFull(F, kernel, kernel_bare, kgrid, qgrids, cm)
    delta0, delta2 = calcΔ(Ft,  kernel_t, kernel_bare, fdlr, kgrid, qgrids)
    delta_new2 = real(DLR.tau2matfreq(:acorr, delta2, fdlr, fdlr.n, axis=2))
    for ni in 1:fdlr.size
        delta_new2[:, ni] += delta0[:]
    end

    println(delta_new3[kF_label,:])
    println(delta_new1[kF_label,:])
    println(delta_new2[kF_label,:])
    println(delta_new1[kF_label,:] ./ delta_new2[kF_label,:])
    return delta, F
end

function manualΔ_freq(F_low, F_high, kernel_freq, kernel_bare, kgrid, qgrids, cm)
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
    println("max F:$(maximum(abs.(real(F_low)))), $(maximum(abs.(real(F_high))))")
    println("max dlr coef:$(maximum(abs.(real(Fl_dlr)))), $(maximum(abs.(real(Fh_dlr))))")
    println("max dlr imag:$(maximum(abs.(imag(Fl_dlr)))), $(maximum(abs.(imag(Fh_dlr))))")

    kernel_t = real(DLR.matfreq2tau(:corr, kernel_freq, bdlr, fdlr.τ, axis=3))

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
                fxl = @view F_low[head:tail, mi] # all F in the same kpidx-th K panel
                FFl[mi] = barycheb(order, q, fxl, w, x) # the interpolation is independent with the panel length
                fxh = @view F_high[head:tail, mi] # all F in the same kpidx-th K panel
                FFh[mi] = barycheb(order, q, fxh, w, x) # the interpolation is independent with the panel length
            end
            FFl = real(DLR.matfreq2dlr(:acorr, FFl, fdlr, axis=1))
            FFh = real(DLR.matfreq2dlr(:acorr, FFh, fdlr, axis=1))

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

function test_calcΔ_sep(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
    kF_label = searchsortedfirst(kgrid.grid, kF)
    qF_label = searchsortedfirst(qgrids[kF_label].grid, kF)

    n_c = Base.floor(Int,Ω_c/(2π)*β)
    println("Sep:$(Ω_c), n_c:$(n_c)")
    cm = FreqConv.ConvMat(bdlr, fdlr, n_c)


    kernel_t = real(DLR.matfreq2tau(:corr, kernel_freq, bdlr, fdlr.τ, axis=3))
    kernel_dlr = (DLR.matfreq2dlr(:corr, kernel_freq, bdlr, axis=3))
    println("compare kernel dlr")
    println(real(kernel_dlr[qF_label,kF_label,:]))
    println(real(DLR.matfreq2dlr(:corr, kernel_freq[kF_label,qF_label,:], bdlr, axis=1)))
    #kernel = zeros(Float64, (size(kernel_freq)[1], size(kernel_freq)[2], fdlr.size, fdlr.size))
    #kernel_t = 0.0 .* kernel_t

    F = calcF_freqSep(delta, Σ, fdlr, kgrid)
    Ft = DLR.matfreq2tau(:acorr, F, fdlr, fdlr.τ, axis=2)
    F_dlr = DLR.matfreq2dlr(:acorr, F, fdlr, axis=2)

    kernel = kernel_sep_full(kernel_freq, cm)
    kernel_low, kernel_high = kernel_sep(kernel_freq, cm)
    #kernel = DLR.matfreq2dlr(:corr, kernel_freq, bdlr, axis=3)

    delta_new3 = calcΔ_freqSep(F, F .* 0.0, kernel_low, kernel_high, kernel_bare, kgrid, qgrids, cm)
    delta_new2 = manualΔ_freq(F, F .* 0.0, kernel_freq , kernel_bare, kgrid, qgrids, cm)

    println(delta_new3[kF_label,:])
    println(delta_new2[kF_label,:])
    println(delta_new3[kF_label,:] ./ delta_new2[kF_label,:])
    return delta, F
end


if abspath(PROGRAM_FILE) == @__FILE__
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    outFileName = rundir*"/flow_$(WID).dat"
    if !(isfile(outFileName))
        f = open(outFileName, "a")
        @printf(f, "%.6e\t%.6f\t%.6e\t%d\n", β, rs, mom_sep, channel)
        close(f)
    end    
    fdlr = DLR.DLRGrid(:acorr, fEUV, β, 1e-10)
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
    #kernel_bare = kernel_bare .* 0.0
    println(kernel_freq[kF_label,qF_label,:])

    kernel = real(DLR.matfreq2tau(:corr, kernel_freq, bdlr, fdlr.τ, axis=3))
    println(size(kernel_freq),size(kernel))

    #err test section
    kpanel2 =  KPanel(Nk, kF, maxK, minK)
    kgrid_double = CompositeGrid(kpanel2, 2*order, :cheb)
    qgrids_double = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), 2*order, :gaussian) for k in kgrid_double.grid]
    
    #initialize delta
    delta = delta_init(fdlr, kgrid)
    #delta = zeros(Float64, (length(kgrid.grid), fdlr.size)) .+ 1.0
    delta_0 = zeros(Float64, length(kgrid.grid)) .+ 1.0

    #Σ = (0.0+0.0im) * delta
    #test_calcΔ(delta, kernel_freq, kernel_bare .* 0.0, Σ, kgrid, qgrids, fdlr, bdlr)
    #@assert 1==2 "end"

    if(sigma_type == :none)
        Σ = (0.0+0.0im) * delta
        if method_type == :explicit
            Δ , F, lamu = Explicit_Freq(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
        else
            #Δ , F, lamu = Explicit_Freq(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
            Δ_final_low, Δ_final_high,  F, lamu = Implicit_Renorm_Freq(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
        end
    else
        w0_label = 0
        dataFileName = rundir*"/sigma_$(WID).dat"
        f = open(dataFileName, "r")
        Σ_raw = readdlm(f)
        Σ  = transpose(reshape(Σ_raw[:,1],(fdlr.size,length(kgrid.grid)))) + transpose(reshape(Σ_raw[:,2],(fdlr.size,length(kgrid.grid))))*im
        if method_type == :explicit
            Δ , F, lamu = Explicit_Freq(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
        else
            Δ_final_low, Δ_final_high,  F, lamu = Implicit_Renorm_Freq(delta, kernel_freq, kernel_bare, Σ, kgrid, qgrids, fdlr, bdlr)
        end
    end

    if method_type == :implicit
        Δ_out = lamu .* Δ_final_low .+ Δ_final_high
        #F_τ = DLR.matfreq2dlr(:acorr, F, fdlr, axis=2)
        #F_τ = real.(DLR.dlr2tau(:acorr, F_τ, fdlr, extT_grid.grid , axis=2))
        #F_τ = real(DLR.matfreq2tau(:acorr, F_freq, fdlr, extT_grid.grid, axis=2))
        #println("F_τ_max:",maximum(F_τ))
        #F_ext = zeros(Float64, (length(extK_grid.grid), length(extT_grid.grid)))

        
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
