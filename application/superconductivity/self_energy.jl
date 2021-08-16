using StaticArrays:similar, maximum
using QuantumStatistics: Grid, FastMath, Utility
using Lehmann
using Printf
using Plots
using DelimitedFiles

srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

if !(@isdefined β)
    include(rundir*"/parameter.jl")
    using .parameter
end

include("eigen.jl")
include("grid.jl")

function KO_sigma(q, n)
    g = e0^2
    kernal = 0.0
    G_s=A1*q^2/(1.0+B1*q^2)+A2*q^2/(1.0+B2*q^2);
    G_a=A1*q^2/(1.0+B1*q^2)-A2*q^2/(1.0+B2*q^2);
    spin_factor=3.0
    if abs(q) > EPS 
        x = q/2/kF
        ω_n = 2*π*n/β
        y = me*ω_n/q/kF
        
        
        if n == 0
            if abs(q - 2*kF) > EPS
                Π = me*kF/2/π^2*(1 + (1 -x^2)*log1p(4*x/((1-x)^2))/4/x)
            else
                Π = me*kF/2/π^2
            end
            kernal = - Π*(1-G_s)^2/( q^2/4/π/g  + Π*(1-G_s))-spin_factor*Π*(-G_a)^2/(q^2/4/π/g + Π*(-G_a))
        else
            if abs(q - 2*kF) > EPS
                theta = atan( 2*y/(y^2+x^2-1) )
                if theta < 0
                    theta = theta + π
                end
                @assert theta >= 0 && theta<= π
                Π = me*kF/2/π^2*(1 + (1 -x^2 + y^2)*log1p(4*x/((1-x)^2+y^2))/4/x - y*theta)                       
            else
                theta = atan( 2/y )
                if theta < 0
                    theta = theta + π
                end
                @assert theta >= 0 && theta<= π
                Π = me*kF/2/π^2*(1 + y^2*log1p(4/y^2)/4 - y*theta)                       
            end
            Π0 = Π / q^2
            #kernal = - Π0/( 1.0/4/π/g  + Π0 )
            kernal = - Π0*(1-G_s)^2/( 1.0/4/π/g  + Π0*(1-G_s))-spin_factor*Π0*(-G_a)^2/(1.0/4/π/g + Π0*(-G_a))
            #kernal = - Π/( (q^2+mass2)/4/π/g  + Π )

        end
       
        #kernal = Π
    else
        kernal = 0
        
    end

    return kernal
end


function Composite_int_sigma(k, p, n, grid_int)
    sum = 0
    sum_bare = 0
    g = e0^2

    if interaction_type==:rpa
        W_DYNAMIC=RPA
    elseif interaction_type==:ko
        W_DYNAMIC=KO_sigma
    end

    for (qi, q) in enumerate(grid_int.grid)
        legendre_x = (k^2 + p^2 - q^2)/2/k/p
        if(abs(abs(legendre_x)-1)<1e-12)
            legendre_x = sign(legendre_x)*1
        end
        wq = grid_int.wgrid[qi]
        sum += Pl(legendre_x, channel)*4*π*g/q*W_DYNAMIC(q, n) * wq
        sum_bare += Pl(legendre_x, channel)*4*π*g/q * wq
    end
    return sum_bare, sum
end

function legendre_dc_sigma(bdlr, kgrid, qgrids, kpanel_bose, int_order)
    kernal_bare = zeros(Float64, (length(kgrid.grid), length(qgrids[1].grid)))
    kernal = zeros(Float64, (length(kgrid.grid), length(qgrids[1].grid), bdlr.size))
    for (ki, k) in enumerate(kgrid.grid)
        for (pi, p) in enumerate(qgrids[ki].grid)
            for (ni, n) in enumerate(bdlr.n)
                if abs(k - p) > EPS
                    grid_int = build_int(k, p ,kpanel_bose, int_order)
                    kernal_bare[ki,pi], kernal[ki,pi,ni] = Composite_int_sigma(k, p, n, grid_int)
                    @assert isfinite(kernal[ki,pi,ni]) "fail kernal at $ki,$pi,$ni, with $(kernal[ki,pi,ni])"
                else
                    kernal_bare[ki,pi] = 0
                    kernal[ki,pi,ni] = 0
                end
            end
        end
    end
    
    return kernal_bare,  kernal
end


function G0_τ(k, τ)
    ω = k^2 / (2me) - EF
    return Spectral.kernelFermiT(τ, ω, β)
end

function G0_ω(k, ωn)
    ω = k^2 / (2me) - EF
    return Spectral.kernelFermiT(τ, ω, β)
end

function G_τ(ΣR, ΣI, fdlr, kgrid)
    G = zeros(ComplexF64, (length(kgrid.grid), fdlr.size))

    for (ki, k) in enumerate(kgrid.grid)
        for (ωni, ωn) in enumerate(fdlr.ωn)
            ω = k^2 / (2me) - EF
            G[ki,ωni] = 1/( im*ωn -ω + ΣR[ki,ωni] + im*ΣI[ki,ωni] )
        end
    end

    G = DLR.matfreq2tau(:fermi, G, fdlr, fdlr.τ, axis=2)

    return G
end

function calcΣ(G, kernal, kernal_bare, fdlr, kgrid, qgrids)
    
    Σ0 = zeros(ComplexF64, length(kgrid.grid))
    Σ = zeros(ComplexF64, (length(kgrid.grid), fdlr.size))
    order = kgrid.order

    G_ins = DLR.tau2dlr(:fermi, G, fdlr, axis=2)
    G_ins = -DLR.dlr2tau(:fermi, G_ins, fdlr, [β-EPS,] , axis=2)[:,1]

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
                fx = @view G[head:tail, τi] # all F in the same kpidx-th K panel
                FF = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length

                wq = qgrids[ki].wgrid[qi]
                #Δ[ki, τi] += dH1(k, q, τ) * FF * wq
                Σ[ki, τi] += kernal[ki ,qi ,τi] * FF * wq /k * q
                @assert isfinite(Σ[ki, τi]) "fail Δ at $ki, $τi with $(Δ[ki, τi]), $FF\n $q for $fx\n $x \n $w\n $q < $(kgrid.panel[kpidx + 1])"

                if τi == 1
                    fx_ins = @view G_ins[head:tail] # all F in the same kpidx-th K panel
                    FF = barycheb(order, q, fx_ins, w, x) # the interpolation is independent with the panel length

                    #Δ0[ki] += bare(k, q) * FF * wq
                    Σ0[ki] += kernal_bare[ki, qi] * FF * wq /k * q
                    @assert isfinite(Σ0[ki]) "fail Δ0 at $ki with $(Δ0[ki])"
                end

            end
        end
    end
    
    return Σ0./(4*π^2), Σ./(4*π^2)
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


function check_Σ0(filename)
    f = open(filename, "r")
    data = readdlm(f)
    raw_mom = data[1:400,1]
    raw_Σ0 = data[1:400,2]

    kgrid, Σ0 = main_G0W0(false)

    Σ0_compare = interpolate(Σ0, kgrid, raw_mom)

    println(maximum(abs.(Σ0_compare-raw_Σ0)))

    pic1 = plot(raw_mom, Σ0_compare)
    plot!(raw_mom, raw_Σ0)
    display(pic1)
    #savefig(pic1, "kl_escale.pdf")
    readline()

end


function main_G0W0(istest=false)
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    println(G0_τ(1.0, EPS), "\t", G0_τ(1.0, -EPS))
    fdlr = DLR.DLRGrid(:fermi, ΣEUV, β, 1e-10)	
    bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10) 

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

    kernal_bare, kernal_freq = legendre_dc_sigma(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    kernal = real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, fdlr.τ, axis=3))
    println(typeof(kernal))

    Σ0, Σ = calcΣ(kernal, kernal_bare, fdlr, kgrid, qgrids)

    Σ_freq = DLR.tau2matfreq(:fermi, Σ, fdlr, fdlr.n, axis=2)
    Σ_compare = DLR.matfreq2tau(:fermi, Σ_freq, fdlr, fdlr.τ, axis=2)
    println("$(maximum(abs.(real.(Σ_compare) - Σ))),$(maximum(abs.(imag(Σ_compare))))")
    ΣR = real(Σ_freq)
    ΣI = imag(Σ_freq)

    Σ_shift = ΣR[kF_label,ω0_label]+Σ0[kF_label]

    if istest
        n1, n2 = 31, 30
        println(fdlr.n)
        println(Σ0)
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

    return kgrid, Σ0
end

function main_GW0(istest=false)
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    println(G0_τ(1.0, EPS), "\t", G0_τ(1.0, -EPS))
    fdlr = DLR.DLRGrid(:fermi, ΣEUV, β, 1e-10)
    bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10)

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

    kernal_bare, kernal_freq = legendre_dc_sigma(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    kernal = real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, fdlr.τ, axis=3))
    println(typeof(kernal))

    Σ0, Σ = calcΣ(kernal, kernal_bare, fdlr, kgrid, qgrids)
    Σ_freq = DLR.tau2matfreq(:fermi, Σ, fdlr, fdlr.n, axis=2)
    ΣR = real(Σ_freq)
    ΣI = imag(Σ_freq)
    Σ_shift = ΣR[kF_label,ω0_label]+real(Σ0)[kF_label]
    ΣR = broadcast(+, real(Σ0) .- Σ_shift, ΣR)
    ΣI = broadcast(+, imag(Σ0), ΣI)
    Σ_shift_old = Σ_shift
    for i in 1:100
        Gt = G_τ(ΣR, ΣI, fdlr, kgrid)
        Σ0, Σ = calcΣ(Gt, kernal, kernal_bare, fdlr, kgrid, qgrids)
        Σ_freq = DLR.tau2matfreq(:fermi, Σ, fdlr, fdlr.n, axis=2)
        ΣR = real(Σ_freq)
        ΣI = imag(Σ_freq)
        if istest
            Σ_tau_compare = DLR.matfreq2tau(:fermi, Σ_freq, fdlr, fdlr.τ, axis=2)
            Σ_freq_compare = DLR.tau2matfreq(:fermi, Σ_tau_compare, fdlr, fdlr.n, axis=2)
            println(maximum(abs.(real(Σ_freq_compare)-ΣR)))
            Σ_dlr = DLR.tau2dlr(:fermi, Σ, fdlr, axis=2)

            pic = plot(fdlr.ω, real(Σ_dlr)[kF_label,:])
            display(pic)
            readline()
        end
        Σ_shift = ΣR[kF_label,ω0_label]+real(Σ0)[kF_label]
        ΣR = broadcast(+, real(Σ0) .- Σ_shift, ΣR)
        ΣI = broadcast(+, imag(Σ0), ΣI)
        println(Σ_shift)
        if abs((Σ_shift-Σ_shift_old)/Σ_shift)<1e-10
            break
        end
        Σ_shift_old = Σ_shift
    end

    if istest
        n1, n2 = 31, 30
        println(fdlr.n)
        println(ΣR[kF_label,:])
        println(ΣI[kF_label,:])

        pic1 = plot(fdlr.n[n1:end-n2], ΣR[kF_label,n1:end-n2])
        plot!(pic1,fdlr.n[n1:end-n2], ΣI[kF_label,n1:end-n2])
        display(pic1)
        #savefig(pic1, "kl_escale.pdf")
        readline()

        pic2 = plot(fdlr.τ, real(Σ)[kF_label,:])
        display(pic2)
        readline()

        pic3 = plot(kgrid.grid, real(Σ0))
        display(pic3)
        readline()

    end


    outFileName = rundir*"/sigma_$(WID).dat"
    f = open(outFileName, "w")
    for (ki, k) in enumerate(kgrid.grid)
        for (ni, n) in enumerate(fdlr.τ)
            @printf(f, "%32.17g\t%32.17g\n", ΣR[ki,ni], ΣI[ki,ni])
        end
    end


end


if abspath(PROGRAM_FILE) == @__FILE__
    check_Σ0(rundir * "/" * ARGS[2])
    if sigma_type == :gw0
        main_GW0(false)
    elseif sigma_type == :g0w0
        main_G0W0(false)
    end

end

