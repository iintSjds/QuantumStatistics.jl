module FreqConv

using Lehmann, StaticArrays

function fullMat(α, γ, n, β)
    ω=π*(2n+1)/β

    if α==0
        return Spectral.kernelCorrΩ(0, α, β)*Spectral.kernelAnormalCorrΩ(n, γ, β)/β
    end
    factor = 4*α*γ*(-expm1(-α*β))*(1+ℯ^(-γ*β))
    f_term = 1.0/((α^2-γ^2+ω^2)^2+4*γ^2*ω^2)*(α^2-γ^2+ω^2)/(2γ)/(1+ℯ^(-γ*β))*(-expm1(-γ*β))
    b_term = 1.0/((-α^2+γ^2+ω^2)^2+4*α^2*ω^2)*(-α^2+γ^2+ω^2)/(2α)/(-expm1(-α*β))*(1+ℯ^(-α*β))
#    println(factor, "\t", f_term, "\t", b_term)
    return factor*(f_term+b_term)
end

function sumMat(α, γ, n, β, n_c)
    result = 0.0
    for m in 1:n_c
        result += Spectral.kernelAnormalCorrΩ(m, γ, β)*Spectral.kernelCorrΩ(n-m, α, β)
    end

    return result/β
end



struct ConvMat
    bdlr::DLR.DLRGrid
    fdlr::DLR.DLRGrid

    n_c::Int

    full_mat::Array{Float64,3}
    sep_mat::Array{Float64,3}

    function ConvMat(bdlr::DLR.DLRGrid, fdlr::DLR.DLRGrid, n_c::Int)
        sep_mat = zeros(Float64, (fdlr.size, bdlr.size, fdlr.size))
        full_mat = zeros(Float64, (fdlr.size, bdlr.size, fdlr.size))
        β = fdlr.β

        for (ni, n) in enumerate(fdlr.n)
            for (ωi, ω) in enumerate(bdlr.ω)
                for (ξi, ξ) in enumerate(fdlr.ω)
                    # for m in 1:n_c
                    #     sep_mat[ni, ωi, ξi] += Spectral.kernelAnormalCorrΩ(m, ξ, β)*Spectral.kernelCorrΩ(n-m, ω, β)
                    # end
                    sep_mat[ni, ωi, ξi] = sumMat(ω, ξ, n, β, n_c)
                    full_mat[ni, ωi, ξi] = fullMat(ω, ξ, n, β)
                end
            end
        end

        return new(bdlr, fdlr, n_c, full_mat, sep_mat)
    end

end

function freq_conv(Γ::Vector, F::Vector, CM::ConvMat, type=:full)
    @assert length(Γ) == CM.bdlr.size
    @assert length(F) == CM.fdlr.size

    conv_mat = CM.full_mat
    if type == :low
        conv_mat = CM.sep_mat
    elseif type == :high
        conv_mat -= CM.sep_mat
    end

    Δ = zeros(ComplexF64, CM.fdlr.size)

    γ = DLR.matfreq2dlr(:corr, Γ, CM.bdlr)
    f = DLR.matfreq2dlr(:acorr, F, CM.fdlr)

    # println(CM.fdlr.n)
    # println(CM.bdlr.ω)
    # println(CM.fdlr.ω)
    for (ni, n) in enumerate(CM.fdlr.n)
        for (ωi, ω) in enumerate(CM.bdlr.ω)
            for (ξi, ξ) in enumerate(CM.fdlr.ω)
                Δ[ni] += γ[ωi] * f[ξi] * conv_mat[ni, ωi, ξi]
            end
        end
    end

    return real(Δ)
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

end


if abspath(PROGRAM_FILE) == @__FILE__
    const E_c = π
    const Ω_c = 1.0
    const Ω_c2 = π
    const g = 2.0

    const fEUV = 10
    const bEUV = 10

    const β = 200.0

    function Phonon(ω)
        return g .* ω .^ 2 ./ (ω .^ 2 .+ Ω_c^2 ) #.- g .* ω .^ 2 ./ (ω .^ 2 .+ Ω_c2^2 )
    end


    fdlr = FreqConv.DLR.DLRGrid(:acorr, fEUV, β, 1e-12)
    bdlr = FreqConv.DLR.DLRGrid(:corr, bEUV, β, 1e-12)

    println("α:", bdlr.ω)
    println("γ:", fdlr.ω)

    #    α, ξ = bdlr.ω[2],fdlr.ω[3]
    α, ξ = 0.00073, 0.056
    n = 2
    m = 5
    println("sumMat:", FreqConv.sumMat(α,ξ,n,β,100),"\t",FreqConv.sumMat(α,ξ,n,β,1000),"\t",FreqConv.sumMat(α,ξ,n,β,10000),"\t")
    println("fullMat:",FreqConv.fullMat(α,ξ,n,β))
    println(FreqConv.Spectral.kernelCorrΩ(n-m, α, β)*FreqConv.Spectral.kernelAnormalCorrΩ(m, ξ, β))

#    @assert 1==2 "break"

    n_max = Base.floor(Int,E_c/(2π)*β)
    n_c = Base.floor(Int,Ω_c/(2π)*β)
    println("n_c=$(n_c)")
    N = 33

    cm = FreqConv.ConvMat(bdlr, fdlr, n_c)

    # Γ = Phonon(bdlr.ωn)
    # ωf = 2π .* (fdlr.n .+ 0.5) ./ β
    # F = (1.0 .- 0.1 .* Phonon(ωf)) ./ ωf
    # println(Γ)
    # println(F)
    # γ = FreqConv.DLR.matfreq2dlr(:corr, Γ, bdlr)
    # f = FreqConv.DLR.matfreq2dlr(:acorr, F, fdlr)

    Γt = FreqConv.MultiPole(:corr, bdlr.τ, β, bEUV)[1]
    Ft = FreqConv.MultiPole(:acorr, fdlr.τ, β, fEUV)[1]
    γ = FreqConv.DLR.tau2dlr(:corr, Γt, bdlr)
    f = FreqConv.DLR.tau2dlr(:acorr, Ft, fdlr)
    Γ = FreqConv.DLR.dlr2matfreq(:corr, γ, bdlr, bdlr.n)
    F = FreqConv.DLR.dlr2matfreq(:acorr, f, fdlr, fdlr.n)


    Δ = FreqConv.freq_conv(Γ, F, cm, :low)
    println(fdlr.n[1:N])
    println(Δ[1:N])

    Δ_comp = zeros(Float64, fdlr.size)
    Γ_mat = zeros(Float64, (fdlr.size, n_c))
    F_v = zeros(Float64, n_c)

    for (ni, n) in enumerate(fdlr.n)
        Γ_mat[ni, :] = FreqConv.DLR.dlr2matfreq(:corr, γ, bdlr, [n-m for m in 1:n_c])
    end
    F_v = FreqConv.DLR.dlr2matfreq(:acorr, f, fdlr, [m for m in 1:n_c])

    for (ni, n) in enumerate(fdlr.n)
        for m in 1:n_c
            # Δ_comp[ni] += test_spec(2π*(n-m)/β) * test_spec(2π*(m+0.5)/β)
            Δ_comp[ni] += Γ_mat[ni, m]*F_v[m]
        end
    end

    println(Δ_comp[1:N])

    cm = FreqConv.ConvMat(bdlr, fdlr, 10000)

    Δ = FreqConv.freq_conv(Γ, F, cm, :full)
    println(Δ)
    Γft = FreqConv.DLR.dlr2tau(:corr, γ, bdlr, fdlr.τ)
    Δ2t = Γft .* Ft
    Δ2 = FreqConv.DLR.tau2matfreq(:acorr, Δ2t, fdlr, fdlr.n)
    println(real(Δ2))
end
