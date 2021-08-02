using StaticArrays:similar, maximum
using QuantumStatistics
using Printf
using Plots

srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

if !(@isdefined β)
    include(rundir*"/parameter.jl")
    using .parameter
end

include("eigen.jl")
include("grid.jl")

function G0_τ(k, τ)
    ω = k^2 / (2me) - EF
    return Spectral.kernelFermiT(τ, ω, β)
end

function G0_ω(k, ωn)
    ω = k^2 / (2me) - EF
    return Spectral.kernelFermiT(τ, ω, β)
end


function calcΣ(kernal, kernal_bare, fdlr, kgrid, qgrids)
    
    Σ0 = zeros(Float64, length(kgrid.grid))
    Σ = zeros(Float64, (length(kgrid.grid), fdlr.size))
    order = kgrid.order

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

            
            for (τi, τ) in enumerate(fdlr.τ)

                FF = G0_τ(q, τ)
                wq = qgrids[ki].wgrid[qi]
                #Δ[ki, τi] += dH1(k, q, τ) * FF * wq
                Σ[ki, τi] += kernal[ki ,qi ,τi] * FF * wq /k * q
                @assert isfinite(Σ[ki, τi]) "fail Δ at $ki, $τi with $(Δ[ki, τi]), $FF\n $q for $fx\n $x \n $w\n $q < $(kgrid.panel[kpidx + 1])"

                if τi == 1
                    FF = G0_τ(q, -EPS)
                    #Δ0[ki] += bare(k, q) * FF * wq
                    Σ0[ki] += kernal_bare[ki, qi] * FF * wq /k * q
                    @assert isfinite(Σ0[ki]) "fail Δ0 at $ki with $(Δ0[ki])"
                end

            end
        end
    end
    
    return Σ0./(4*π^2), Σ./(4*π^2)
end


function main(istest=false)
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    println(G0_τ(1.0, EPS), "\t", G0_τ(1.0, -EPS))
    if(β<=10000)
    	  fdlr = DLR.DLRGrid(:fermi, 1000EF, β, 1e-10)
    else
	      fdlr = DLR.DLRGrid(:fermi, 100EF, β, 1e-10)	
    end
    bdlr = DLR.DLRGrid(:corr, 100EF, β, 1e-10) 

    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    # kgrid2 = CompositeGrid(kpanel, order÷2, :cheb)
    # qgrids2 = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order÷2, :gaussian) for k in kgrid.grid] # qgrid for each k
    kF_label = searchsortedfirst(kgrid.grid, kF)
    ω0_label = searchsortedfirst(fdlr.n, 0)
    println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    kernal_bare, kernal_freq = legendre_dc(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    kernal = real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, fdlr.τ, axis=3))
    println(typeof(kernal))

    Σ0, Σ = calcΣ(kernal, kernal_bare, fdlr, kgrid, qgrids)

    Σ_freq = DLR.tau2matfreq(:fermi, Σ, fdlr, fdlr.n, axis=2)
    ΣR = real(Σ_freq)
    ΣI = imag(Σ_freq)

    Σ_shift = ΣR[kF_label,ω0_label]+Σ0[kF_label]

    if istest
        n1, n2 = 31, 30
        println(fdlr.n)
        println(ΣR[kF_label,:].+Σ0[kF_label].-Σ_shift)
        println(ΣI[kF_label,:])
        println(Σ[kF_label,:])

        pic1 = plot(fdlr.n[n1:end-n2], ΣR[kF_label,n1:end-n2].+Σ0[kF_label].-Σ_shift)
        plot!(pic1,fdlr.n[n1:end-n2], ΣI[kF_label,n1:end-n2])
        display(pic1)
        #savefig(pic1, "kl_escale.pdf")
        readline()

        pic2 = plot(fdlr.τ, Σ[kF_label,:])
        display(pic2)
        readline()

        pic3 = plot(kgrid.grid, Σ0)
        display(pic3)
        readline()

    end


    outFileName = rundir*"/sigma_$(WID).dat"
    f = open(outFileName, "w")
    for (ki, k) in enumerate(kgrid.grid)
        for (ni, n) in enumerate(fdlr.τ)
            @printf(f, "%32.17g\t%32.17g\n", ΣR[ki,ni]+Σ0[ki]-Σ_shift, ΣI[ki,ni])
        end
    end


end

if abspath(PROGRAM_FILE) == @__FILE__
    main(false)
end

