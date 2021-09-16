
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

function SemiCircle(type, Grid, β, Euv; IsMatFreq=false)
    # calculate Green's function defined by the spectral density
    # S(ω) = sqrt(1 - (ω / Euv)^2) / Euv # semicircle -1<ω<1

    ##### Panels endpoints for composite quadrature rule ###
    npo = Int(ceil(log(β*Euv) / log(2.0)))
    pbp = zeros(Float64, 2npo + 1)
    pbp[npo + 1] = 0.0
    for i in 1:npo
        pbp[npo + i + 1] = 1.0 / 2^(npo - i)
    end
    pbp[1:npo] = -pbp[2npo + 1:-1:npo + 2]

    function Green(n, IsMatFreq)
        #n: polynomial order
        xl, wl = gausslegendre(n)
        xj, wj = gaussjacobi(n, 1 / 2, 0.0)

        G = IsMatFreq ? zeros(ComplexF64, length(Grid)) : zeros(Float64, length(Grid))
        err=zeros(Float64, length(Grid))
        for (τi, τ) in enumerate(Grid)
            for ii in 2:2npo-1
                a, b = pbp[ii], pbp[ii+1]
                for jj in 1:n
                    x = (a+b)/2+(b-a)/2*xl[jj]
                    if (type==:corr ||type==:acorr) && x<0.0 
                        #spectral density is defined for positivie frequency only for correlation functions
                        continue
                    end
                    ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv*x, β) : Spectral.kernelT(type, τ, Euv*x, β)
                    G[τi] += (b-a)/2*wl[jj]*ker*sqrt(1-x^2)
                end
            end
        
            a, b = 1.0/2, 1.0
            for jj in 1:n
                x = (a+b)/2+(b-a)/2*xj[jj]
                ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv*x, β) : Spectral.kernelT(type, τ, Euv*x, β)
                G[τi] += ((b-a)/2)^1.5*wj[jj]*ker*sqrt(1+x)
            end

            if type != :corr && type !=:acorr
                #spectral density is defined for positivie frequency only for correlation functions
                a, b = -1.0, -1.0/2
                for jj in 1:n
                    x = (a+b)/2+(b-a)/2*(-xj[n-jj+1])
                    ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv*x, β) : Spectral.kernelT(type, τ, Euv*x, β)
                    G[τi] += ((b-a)/2)^1.5*wj[n-jj+1]*ker*sqrt(1-x)
                end
            end
        end
        return G
    end

    g1=Green(24, IsMatFreq)
    g2=Green(48, IsMatFreq)
    err=abs.(g1-g2)
    
    println("Semi-circle case integration error = ", maximum(err))
    return g2, err
end

function MultiPole(type, Grid, β, Euv; IsMatFreq=false)
    poles=[-Euv, -0.2*Euv, 0.0, 0.8*Euv, Euv]
    # poles=[0.8Euv, 1.0Euv]
    # poles = [0.0]
    g = IsMatFreq ? zeros(ComplexF64, length(Grid)) : zeros(Float64, length(Grid))
    for (τi, τ) in enumerate(Grid)
        for ω in poles

            if (type==:corr || type==:acorr) && ω<0.0 
                #spectral density is defined for positivie frequency only for correlation functions
                continue
            end
            
            if IsMatFreq==false
                g[τi] += Spectral.kernelT(type, τ, ω, β)
            else
                g[τi] += Spectral.kernelΩ(type, τ, ω, β)
            end
        end
    end
    return g, zeros(Float64, length(Grid))
end

function test_spec(ω)
    return ω^2/(ω^2+1.0)
end


if abspath(PROGRAM_FILE) == @__FILE__
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")

    fdlr = DLR.DLRGrid(:acorr, fEUV, β, 1e-10)
    bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10)

    n_c = 1000
    N = 33

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

    # Γ = SemiCircle(:corr, bdlr.τ, β, bEUV)[1]
    # F = SemiCircle(:acorr, fdlr.τ, β, fEUV)[1]
    Γ = MultiPole(:corr, bdlr.τ, β, bEUV)[1]
    F = MultiPole(:acorr, fdlr.τ, β, fEUV)[1]
    γ = DLR.tau2dlr(:corr, Γ, bdlr)
    f = DLR.tau2dlr(:acorr, F, fdlr)
    Γ = DLR.dlr2matfreq(:corr, γ, bdlr, bdlr.n)
    F = DLR.dlr2matfreq(:corr, f, fdlr, fdlr.n)

    # Γ = bdlr.ωn .^ 2 ./ ( bdlr.ωn .^ 2 .+ 1.0)
    # F = fdlr.ωn .^ 2 ./ ( fdlr.ωn .^ 2 .+ 1.0)
    # γ = DLR.matfreq2dlr(:corr, Γ, bdlr)
    # f = DLR.matfreq2dlr(:acorr, F, fdlr)

    println(maximum(real(γ)), "\t",maximum(real(f)))

    Δ = zeros(Float64, fdlr.size)
    for (ni, n) in enumerate(fdlr.n)
        for (ωi, ω) in enumerate(bdlr.ω)
            for (ξi, ξ) in enumerate(fdlr.ω)
                Δ[ni] += γ[ωi] * f[ξi] * conv_mat[ni, ωi, ξi]
            end
        end
    end
    println(fdlr.n[1:N])
    println(Δ[1:N])

    Δ_comp = zeros(Float64, fdlr.size)
    Γ_mat = zeros(Float64, (fdlr.size, n_c))
    F_v = zeros(Float64, n_c)

    for (ni, n) in enumerate(fdlr.n)
        Γ_mat[ni, :] = DLR.dlr2matfreq(:corr, γ, bdlr, [n-m for m in 1:n_c])
    end
    F_v = DLR.dlr2matfreq(:acorr, f, fdlr, [m for m in 1:n_c])

    for (ni, n) in enumerate(fdlr.n)
        for m in 1:n_c
            # Δ_comp[ni] += test_spec(2π*(n-m)/β) * test_spec(2π*(m+0.5)/β)
            Δ_comp[ni] += Γ_mat[ni, m]*F_v[m]
        end
    end

    println(Δ_comp[1:N])
end
