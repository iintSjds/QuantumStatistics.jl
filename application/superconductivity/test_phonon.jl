
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

const E_c = π
const Ω_c = 1.0
const Ω_c2 = π
const g = 2.0

function Phonon(ω)
    return g .* ω .^ 2 ./ (ω .^ 2 .+ Ω_c^2 ) #.- g .* ω .^ 2 ./ (ω .^ 2 .+ Ω_c2^2 )
end


if abspath(PROGRAM_FILE) == @__FILE__
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")

    fdlr = DLR.DLRGrid(:acorr, fEUV, β, 1e-10)
    bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10)

    n_max = Base.floor(Int,E_c/(2π)*β)
    n_c = Base.floor(Int,Ω_c/(2π)*β)
    println("n_c=$(n_c)")
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

    Γ = Phonon(bdlr.ωn)
    ωf = 2π .* (fdlr.n .+ 0.5) ./ β
    F = (1.0 .- 0.1 .* Phonon(ωf)) ./ ωf
    println(Γ)
    println(F)
    # γ = DLR.tau2dlr(:corr, Γ, bdlr)
    # f = DLR.tau2dlr(:acorr, F, fdlr)
    γ = DLR.matfreq2dlr(:corr, Γ, bdlr)
    f = DLR.matfreq2dlr(:acorr, F, fdlr)
    # Γ = DLR.dlr2matfreq(:corr, γ, bdlr, bdlr.n)
    # F = DLR.dlr2matfreq(:corr, f, fdlr, fdlr.n)

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
