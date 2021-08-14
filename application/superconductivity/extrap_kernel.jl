"""
Power method, damp interation and implicit renormalization
"""
#module eigensolver
using QuantumStatistics: Grid, FastMath, Utility
using Lehmann
using LinearAlgebra
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

function main()

    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "a")
    @printf(f, "%.6f\t%.6e\t%d\n", rs, mom_sep, channel)
    close(f)
    println(DLR)
    # if(β<=10000)
    # 	fdlr = DLR.DLRGrid(:acorr, 1000EF, β, 1e-10)
    # else
	  #     fdlr = DLR.DLRGrid(:acorr, 100EF, β, 1e-10)	
    # end
    fdlr = DLR.DLRGrid(:acorr, 1e8/β, β, 1e-10)
    #fdlr2 = DLR.DLRGrid(:acorr, 100EF, β, 1e-10)
    bdlr_aim = DLR.DLRGrid(:corr, 1e8/β, β, 1e-10)
    bdlr_low = DLR.DLRGrid(:corr, 100EF, 1000000, 1e-10)

    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    # kgrid2 = CompositeGrid(kpanel, order÷2, :cheb)
    # qgrids2 = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order÷2, :gaussian) for k in kgrid.grid] # qgrid for each k
    kF_label = searchsortedfirst(kgrid.grid, kF)
    qF_label = searchsortedfirst(qgrids[kF_label].grid, kF)
    println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    kernel_bare, kernel_freq = legendre_dc(bdlr_aim, kgrid, qgrids, kpanel_bose, order_int)
    kernel = real(DLR.matfreq2tau(:corr, kernel_freq, bdlr_aim, fdlr.τ, axis=3))

    kernel_bare_low, kernel_freq_low = legendre_dc(bdlr_low, kgrid, qgrids, kpanel_bose, order_int)
    kernel_compare = real(DLR.matfreq2dlr(:corr, kernel_freq_low, bdlr_low, axis=3))

    # kernel_compare = real(DLR.matfreq2tau(:corr, kernel_freq_low, bdlr_low, fdlr.τ .* (β/bdlr_low.β)^0, axis=3))
    # println(maximum(abs.(kernel .- kernel_compare)))
    # println(maximum(abs.((kernel .- kernel_compare)./kernel)))
    # println(maximum(abs.(kernel)))
    # pic = plot(fdlr.τ, kernel[kF_label,qF_label,:]) 
    # plot!(pic, fdlr.τ, kernel_compare[kF_label,qF_label,:]) 

    kernel_compare = real(DLR.dlr2matfreq(:corr, kernel_compare, bdlr_low, bdlr_aim.n ./ (β/bdlr_low.β), axis=3))
    println(maximum(abs.(kernel_freq .- kernel_compare)))
    println(maximum(abs.((kernel_freq .- kernel_compare)./kernel_freq)))
    println(maximum(abs.(kernel_freq)))
    pic = plot(bdlr_aim.ωn, kernel_freq[kF_label,qF_label,:], markershape = :o) 
    plot!(pic, bdlr_aim.ωn, kernel_compare[kF_label,qF_label,:], markershape = :x) 
    
    display(pic)
    readline()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
