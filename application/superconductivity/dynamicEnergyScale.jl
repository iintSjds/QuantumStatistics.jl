using QuantumStatistics: σx, σy, σz, σ0, Grid,FastMath, Utility, TwoPoint#, DLR, Spectral
#import Lehmann
using Lehmann
using LinearAlgebra
using DelimitedFiles
using Printf
#using Gaston
using Plots
using Statistics
using LsqFit
using LaTeXStrings
srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

if !(@isdefined β)
    include(rundir*"/parameter.jl")
    using .parameter
end
include("eigensolver.jl")
include("eigen.jl")
include("grid.jl")

linreg(x, y) = hcat(fill!(similar(x), 1), x) \ y



if abspath(PROGRAM_FILE) == @__FILE__
    fdlr = DLR.DLRGrid(:acorr, fEUV, β, 1e-10)
    bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10) 
    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid]


    const kF_label = searchsortedfirst(kgrid.grid, kF)
    #const freq0_label = 22
    const freq0_label = 16
    ifreqlist = [i for i in 1:length(fdlr.n) if i%8==1]

    dataFileName = rundir*"/delta_$(WID).dat"

    f = open(dataFileName, "r")
    Δ_data = readdlm(f)
    raw_0low = Δ_data[:,1]
    raw_0high = Δ_data[:,2]
    raw_low = Δ_data[:,3]
    raw_high = Δ_data[:,4]
    Δ0_low  = transpose(reshape(raw_0low,(fdlr.size,length(kgrid.grid))))
    Δ0_high  = transpose(reshape(raw_0high,(fdlr.size,length(kgrid.grid))))
    Δ_low  = transpose(reshape(raw_low,(fdlr.size,length(kgrid.grid))))
    Δ_high  = transpose(reshape(raw_high,(fdlr.size,length(kgrid.grid))))

    flowFileName = rundir*"/flow_$(WID).dat"
    flowData = read(flowFileName, String)
    λ = parse(Float64, split(flowData)[end])

    Δ0 = λ*Δ0_low +Δ0_high
    Δ = λ*Δ_low + Δ_high

    # println(Δ0[kF_label,1:freq0_label])
    # println(Δ[kF_label,1:freq0_label])
    
    Δ_freq = real(DLR.tau2matfreq(:acorr, Δ, fdlr, fdlr.n, axis=2) .+ Δ0)
    Δ_freq = Δ_freq ./ Δ_freq[kF_label,1]
    pic1 = plot(xlabel = L"$k/k_F$", ylabel = "Δ", xlims = (0.0,5.2), legend=:bottomright, grid= false  ,foreground_color_legend = nothing)
    ifreqlist[2] += 5     
    ifreqlist[4] += -5
    ifreqlist[5] += -10
    kmax_label = searchsortedfirst(kgrid.grid, 5.0)
    for i in ifreqlist
        w = π*(2*fdlr.n[i]+1)/β
        if i == ifreqlist[length(ifreqlist)]
            w_st = @sprintf "%.3f" w
        else
            w_st = @sprintf "%.4f" w
        end
        plot!(pic1, kgrid.grid[1:kmax_label], Δ_freq[1:kmax_label, i]./kgrid.grid[1:kmax_label], label = L"$ \omega_n /E_F= %$w_st $")
    end
    savefig(pic1, rundir*"/delta_full.pdf")

    pic2 = plot(xlabel = L"$\omega_n/E_F$", ylabel = L"$\Delta(k_F, \omega_n)$", xlims = (0.0,20), grid= false, legend = false, framestyle = :origin )
    kf_label = searchsortedfirst(kgrid.grid, kF)
    plot!(pic2, (2*fdlr.n[:] .+ 1)*π/β, Δ_freq[kF_label, :])
    savefig(pic2, rundir*"/delta_freq.pdf")
    #readline()
    


    # #const freq0_label = searchsortedfirst(Δ_freq[kF_label, 1:end], 0.0)
    # println(fdlr.ωn[1:freq0_label])
    # println(Δ_freq[kF_label,1:freq0_label])
    

    # #a, b = linreg(fdlr.ωn[1:freq0_label] .^ 2, Δ_freq[kF_label, 1:freq0_label])
    # @. model(x, p) = Δ_freq[kF_label,1] - p[1]*(x-fdlr.ωn[1]) - p[2]*(x-fdlr.ωn[1])^2
    # #@. model(x, p) = Δ_freq[kF_label,1] .- p[1].*log.(x./fdlr.ωn[1]) .- p[2] .* x.^2
    # fit = curve_fit(model,fdlr.ωn[1:freq0_label], Δ_freq[kF_label, 1:freq0_label], [0.0, 100.0] )

    # println(fit.resid)
    # println(coef(fit))
    # a = Δ_freq[kF_label,1]
    # k, b = coef(fit)
    # root = (sqrt(k^2+4*a*b)-k)/2/b
    # #println(a,"\t",b)
    # println(rs, "\t", channel, "\t",root)

    # outFileName = rundir*"/e_scale.dat"
    # f = open(outFileName, "w")
    # @printf(f, "%f %f %32.17g\n",rs,channel, root)
    # close(f)

    # plt=scatter(fdlr.ωn[1:freq0_label],Δ_freq[kF_label,1:freq0_label], label="Δ(k_F,ω_n )", xlabel="ω_n",ylabel="Δ",markershape = :x, legendfontsize=5, title="r_s=$rs, ℓ=$channel, β=$β")
    # #plot!(fdlr.ωn[1:freq0_label],a .- k .*fdlr.ωn[1:freq0_label] .-  b .* fdlr.ωn[1:freq0_label] .^ 2)
    # plot!(fdlr.ωn[1:freq0_label],model(fdlr.ωn[1:freq0_label], coef(fit)), label = "$a - $k ω - $b ω^2")
    # #display(plt)
    # savefig(plt, rundir*"/deltafreq.pdf")
    # #readline()

    # outFileName = rundir*"/delta_freq_kF.dat"
    # f = open(outFileName, "w")
    # for i in 1:length(fdlr.n)
    #     @printf(f, "%f\t%f\n",fdlr.ωn[i],Δ_freq[kF_label,i])
    # end

    # close(f)

end
