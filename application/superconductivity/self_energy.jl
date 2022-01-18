using StaticArrays:similar, maximum
#using QuantumStatistics: Grid, FastMath, Utility
using Lehmann
using Printf
using Plots
using DelimitedFiles
#using LaTeXStrings

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
                if y^2 < 1e-4/EPS                    
                    theta = atan( 2*y/(y^2+x^2-1) )
                    if theta < 0
                        theta = theta + π
                    end
                    @assert theta >= 0 && theta<= π
                    Π = me*kF/2/π^2 * (1 + (1 -x^2 + y^2)*log1p(4*x/((1-x)^2+y^2))/4/x - y*theta)
                else
                    Π = me*kF/2/π^2 * (2.0/3.0/y^2  - 2.0/5.0/y^4) #+ (6.0 - 14.0*(ω_n/4.0)^2)/21.0/y^6)
                end    
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
        if if_electron == 1
            sum += Pl(legendre_x, 0)*4*π*g/q*W_DYNAMIC(q, n) * wq
            sum_bare += Pl(legendre_x, 0)*4*π*g/q * wq
        end
        if if_phonon == 1
            sum += Pl(legendre_x, 0)*q*phonon(q, n) * wq
        end
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
    return Spectral.kernelT(Val(true), Val(:none), τ, ω, β)
end

function G0_ω(k, ωn)
    ω = k^2 / (2me) - EF
    return Spectral.kernelΩ(Val(true), Val(:none), ωn, ω, β)
end

function G0(kgrid, sdlr, adlr)
    GR = zeros(ComplexF64, (length(kgrid.grid), sdlr.size))
    GI = zeros(ComplexF64, (length(kgrid.grid), adlr.size))
    for (ki, k) in enumerate(kgrid.grid)
        for (ωni, ωn) in enumerate(sdlr.ωn)
            ω = k^2 / (2me) - EF
            GR[ki,ωni] = real(-1/( im*ωn -ω ))
        end
    end
    for (ki, k) in enumerate(kgrid.grid)
        for (ωni, ωn) in enumerate(adlr.ωn)
            ω = k^2 / (2me) - EF
            GI[ki,ωni] = imag(-1/( im*ωn -ω ))
        end
    end
    GR = matfreq2tau(sdlr, GR, sdlr.τ, axis=2)
    GI = matfreq2tau(adlr, im*GI, adlr.τ, axis=2)
    return GR, GI
end


function G0(kgrid, fdlr)
    GR = zeros(ComplexF64, (length(kgrid.grid), fdlr.size))
    GI = zeros(ComplexF64, (length(kgrid.grid), fdlr.size))
    G = zeros(ComplexF64, (length(kgrid.grid), fdlr.size))
    for (ki, k) in enumerate(kgrid.grid)
        for (ωni, ωn) in enumerate(fdlr.ωn)
            ω = k^2 / (2me) - EF
            G[ki,ωni] = -1/( im*ωn -ω ) 
        end
    end
    GR = matfreq2tau(fdlr, real(G), fdlr.τ, axis=2)
    GI = matfreq2tau(fdlr, im*imag(G), fdlr.τ, axis=2)
    G =  matfreq2tau(fdlr, G, fdlr.τ, axis=2)
    return G, GR, GI
end
function G_τ(ΣR, ΣI, fdlr, kgrid)
    sumrule = nothing #ones(length(kgrid.grid))
    G = zeros(ComplexF64, (length(kgrid.grid), fdlr.size))

    for (ki, k) in enumerate(kgrid.grid)
        for (ωni, ωn) in enumerate(fdlr.ωn)
            ω = k^2 / (2me) - EF
            G[ki,ωni] = -1/( im*ωn -ω - ΣR[ki,ωni] - im*ΣI[ki,ωni] )
        end
    end

    G = matfreq2tau(fdlr, G, fdlr.τ, axis=2, sumrule = sumrule)

    return G
end

function G_τ_sym(Σ0, ΣR, ΣI, sdlr , adlr, kgrid)

    kF_label = searchsortedfirst(kgrid.grid, kF)
    ω0_label = 1
    GR = zeros(ComplexF64, (length(kgrid.grid), sdlr.size))
    GI = zeros(ComplexF64, (length(kgrid.grid), adlr.size))

    #ΣR_s =  real(matfreq2matfreq(sdlr, ΣR, sdlr.n, axis=2))
    ΣR_s =  real(tau2matfreq(sdlr, ΣR, sdlr.n, axis=2))
    ΣI_s =  imag(tau2matfreq(adlr, ΣI, sdlr.n, axis=2))
    ΣR_a =  real(tau2matfreq(sdlr, ΣR, adlr.n, axis=2))
    #ΣI_a =  imag(matfreq2matfreq(adlr, im*ΣI, adlr.n, axis=2))
    ΣI_a =  imag(tau2matfreq(adlr, ΣI, sdlr.n, axis=2))


    Σ_shift = ΣR_s[kF_label,ω0_label]+real(Σ0)[kF_label]
    ΣR_s = broadcast(+, real(Σ0) .- Σ_shift, ΣR_s)

    Σ_shift = ΣR_a[kF_label,ω0_label]+real(Σ0)[kF_label]
    ΣR_a = broadcast(+, real(Σ0) .- Σ_shift, ΣR_a)
    #println("$(ΣI[kF_label, 1:10])\n")
    ΣI_s = broadcast(+, imag(Σ0), ΣI_s)
    ΣI_a = broadcast(+, imag(Σ0), ΣI_a)
    #println("$(ΣI[kF_label, 1:10])\n")
    for (ki, k) in enumerate(kgrid.grid)
        for (ωni, ωn) in enumerate(sdlr.ωn)
            ω = k^2 / (2me) - EF
            #G[ki,ωni] = -1/( im*ωn -ω - ΣR[ki,ωni] - im*ΣI[ki,ωni] )
            GR[ki,ωni] = (ω + ΣR_s[ki,ωni])/( (ωn - ΣI_s[ki,ωni])^2 + (ω + ΣR_s[ki,ωni])^2 )
        end

    end

    for (ki, k) in enumerate(kgrid.grid)
        for (ωni, ωn) in enumerate(adlr.ωn)
            ω = k^2 / (2me) - EF
            #G[ki,ωni] = -1/( im*ωn -ω - ΣR[ki,ωni] - im*ΣI[ki,ωni] )
            GI[ki,ωni] = im*(ωn - ΣI_a[ki,ωni])/( (ωn - ΣI_a[ki,ωni])^2 + (ω + ΣR_a[ki,ωni])^2 )
        end

    end

    G_real = matfreq2tau(sdlr, GR, sdlr.τ, axis=2)
    G_img = matfreq2tau(adlr, GI, adlr.τ, axis=2)

    return Σ_shift, G_real, G_img
end


function calcΣ(G, kernal, kernal_bare, fdlr, kgrid, qgrids)
    
    Σ0 = zeros(ComplexF64, length(kgrid.grid))
    Σ = zeros(ComplexF64, (length(kgrid.grid), fdlr.size))
    order = kgrid.order

    G_ins = tau2dlr(fdlr, G, axis=2)
    G_ins = -dlr2tau(fdlr, G_ins, [β-EPS,] , axis=2)[:,1]

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


function calcΣ_sym(GR, GI, kernal_s, kernal_a, kernal_bare, sdlr, adlr ,kgrid, qgrids)
    
    Σ0 = zeros(ComplexF64, length(kgrid.grid))
    ΣR = zeros(ComplexF64, (length(kgrid.grid), sdlr.size))
    ΣI = zeros(ComplexF64, (length(kgrid.grid), adlr.size))    
    order = kgrid.order
    #G_ins = tau2dlr(fdlr, G, axis=2)
    #G_ins = -dlr2tau(fdlr, G_ins, [β-EPS,] , axis=2)[:,1]
    G_ins = -tau2tau(sdlr, GR, [β-EPS,] , axis=2)[:,1] - tau2tau(adlr, GI, [β-EPS,] , axis=2)[:,1] 
    for (ki, k) in enumerate(kgrid.grid)        
        kpidx = 1 # panel index of the kgrid
        head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
        x = @view kgrid.grid[head:tail]
        w = @view kgrid.wgrid[head:tail]

        for (qi, q) in enumerate(qgrids[ki].grid)
            wq = qgrids[ki].wgrid[qi]
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

            
            for (τi, τ) in enumerate(sdlr.τ)
                fx = @view GR[head:tail, τi] # all F in the same kpidx-th K panel
                FF = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length

                #Δ[ki, τi] += dH1(k, q, τ) * FF * wq
                ΣR[ki, τi] += kernal_s[ki ,qi ,τi] * FF * wq /k * q
                @assert isfinite(ΣR[ki, τi]) "fail Δ at $ki, $τi with $(Δ[ki, τi]), $FF\n $q for $fx\n $x \n $w\n $q < $(kgrid.panel[kpidx + 1])"

            end
            for (τi, τ) in enumerate(adlr.τ)
                fx = @view GI[head:tail, τi] # all F in the same kpidx-th K panel
                FF = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length
                #Δ[ki, τi] += dH1(k, q, τ) * FF * wq
                ΣI[ki, τi] += kernal_a[ki ,qi ,τi] * FF * wq /k * q
                @assert isfinite(ΣI[ki, τi]) "fail Δ at $ki, $τi with $(Δ[ki, τi]), $FF\n $q for $fx\n $x \n $w\n $q < $(kgrid.panel[kpidx + 1])"

            end
            
            fx_ins = @view G_ins[head:tail] # all F in the same kpidx-th K panel
            FF = barycheb(order, q, fx_ins, w, x) # the interpolation is independent with the panel length
            #Δ0[ki] += bare(k, q) * FF * wq
            Σ0[ki] += kernal_bare[ki, qi] * FF * wq /k * q
            @assert isfinite(Σ0[ki]) "fail Δ0 at $ki with $(Δ0[ki])"
            
        end
    end

    return Σ0./(4*π^2), ΣR./(4*π^2), ΣI./(4*π^2)
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

function main_G0W0(EUV,istest=false)
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    println(G0_τ(1.0, EPS), "\t", G0_τ(1.0, -EPS))
    fdlr = DLRGrid(EUV, β, 1e-10, true, :none)
    adlr = DLRGrid(fEUV, β, 1e-10, true, :pha)
    bdlr = DLRGrid(bEUV, β, 1e-10, false, :ph) 
    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    # kgrid2 = CompositeGrid(kpanel, order÷2, :cheb)
    # qgrids2 = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order÷2, :gaussian) for k in kgrid.grid] # qgrid for each k
    kF_label = searchsortedfirst(kgrid.grid, kF)
    ω0_label = 1 #searchsortedfirst(fdlr.n, 0)
    println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    kernal_bare, kernal_freq = legendre_dc_sigma(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    kernal = real(matfreq2tau(bdlr, kernal_freq, fdlr.τ, axis=3))
    println(typeof(kernal))

    Σ0, Σ = calcΣ(kernal, kernal_bare, fdlr, kgrid, qgrids)
    println(Σ0)
    Σ_freq = tau2matfreq(fdlr, Σ, adlr.n, axis=2)
    # for (ki, k) in enumerate(kgrid.grid)
    #     ω = k^2 / (2me) - EF
    #     for (ni, n) in enumerate(fdlr.n)
    #         np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
    #         nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
    #         println(Spectral.kernelFermiΩ(nn, ω, Σ_freq[ki,ni], β) * Spectral.kernelFermiΩ(np, ω, Σ_freq[ki,ni], β), 1.0/((2*π/β*np-imag(Σ_freq[ki,ni]))^2 + (ω + real(Σ_freq[ki,ni]))^2))
    #         #F[ki, ni] = (Δ[ki, ni]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            
    #     end
    # end

    #Σ_compare = matfreq2tau(fdlr, Σ_freq, fdlr.τ, axis=2)
    #println("$(maximum(abs.(real.(Σ_compare) - Σ))),$(maximum(abs.(imag(Σ_compare))))")
    ΣR = real(Σ_freq)
    ΣI = imag(Σ_freq)

    Σ_shift = ΣR[kF_label,ω0_label]+Σ0[kF_label]
    println(ΣR[kF_label,:].+Σ0[kF_label].-Σ_shift)
    println(ΣI[kF_label,:])
    println(Σ[kF_label,:])
    if istest
        n1, n2 = 31, 30
        println(fdlr.n)
        println(Σ0)
        println(ΣR[kF_label,:].+Σ0[kF_label].-Σ_shift)
        println(ΣI[kF_label,:])
        println(Σ[kF_label,:])
    end
    #     pic1 = plot(fdlr.n[n1:end-n2], ΣR[kF_label,n1:end-n2].+Σ0[kF_label].-Σ_shift)
    #     plot!(pic1,fdlr.n[n1:end-n2], ΣI[kF_label,n1:end-n2])
    #     display(pic1)
    #     #savefig(pic1, "kl_escale.pdf")
    #     readline()

    #     pic2 = plot(fdlr.τ, Σ[kF_label,:])
    #     display(pic2)
    #     readline()

    #     pic3 = plot(kgrid.grid, Σ0)
    #     display(pic3)
    #     readline()

    # end


    outFileName = rundir*"/sigma_$(WID).dat"
    println(outFileName)
    f = open(outFileName, "w")
    for (ki, k) in enumerate(kgrid.grid)
        for (ni, n) in enumerate(adlr.n)
            @printf(f, "%32.17g\t%32.17g\n", ΣR[ki,ni]+Σ0[ki]-Σ_shift, ΣI[ki,ni])
        end
    end
    return Σ_freq

end

function main_GW0(istest=false)
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    println(G0_τ(1.0, EPS), "\t", G0_τ(1.0, -EPS))
    fdlr = DLRGrid( ΣEUV, β, 1e-10, true, :none)
    bdlr = DLRGrid(bEUV, β, 1e-10, false, :ph)
    adlr = DLRGrid(fEUV, β, 1e-10, true, :pha)
 
    println(fdlr.n)    
    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    # kgrid2 = CompositeGrid(kpanel, order÷2, :cheb)
    # qgrids2 = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order÷2, :gaussian) for k in kgrid.grid] # qgrid for each k
    kF_label = searchsortedfirst(kgrid.grid, kF)
    ω0_label = searchsortedfirst(fdlr.n, 0)
    println(fdlr.n[ω0_label])
    ω0_a_label = 1
    println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    kernal_bare, kernal_freq = legendre_dc_sigma(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    kernal = real(matfreq2tau(bdlr, kernal_freq, fdlr.τ, axis=3))
    println(typeof(kernal))

    sumrule = 1.0#nothing#ones(length(kgrid.grid))
    Σ0, Σ = calcΣ(kernal, kernal_bare, fdlr, kgrid, qgrids)
    Σ_freq = tau2matfreq(fdlr, Σ, fdlr.n, axis=2)
    ΣR = real(Σ_freq)
    ΣI = imag(Σ_freq)
    Σ_shift = ΣR[kF_label,ω0_label]+real(Σ0)[kF_label]
    ΣR = broadcast(+, real(Σ0) .- Σ_shift, ΣR)
    ΣI = broadcast(+, imag(Σ0), ΣI)
    Σ_shift_old = Σ_shift
    n1 = ω0_label-24
    n2 = ω0_label+23
    mix = 0.2
    Gt =  G_τ(ΣR, ΣI, fdlr, kgrid)
    for i in 1:1000
        Gt = mix * G_τ(ΣR, ΣI, fdlr, kgrid) + (1-mix) * Gt
        Σ0, Σ = calcΣ(Gt, kernal, kernal_bare, fdlr, kgrid, qgrids)
        Σ_freq = tau2matfreq(fdlr, Σ, fdlr.n, axis=2, sumrule = sumrule)
        ΣR = real(Σ_freq)
        ΣI = imag(Σ_freq)
        println(fdlr.n[n1],"\t",fdlr.n[n2])
        println(ΣR[kF_label,n1],"\t",ΣR[kF_label,n2])
        println(ΣI[kF_label,n1],"\t",ΣI[kF_label,n2])
        if istest
            Σ_tau_compare = matfreq2tau(fdlr, Σ_freq, fdlr.τ, axis=2)
            Σ_freq_compare = tau2matfreq(fdlr, Σ_tau_compare, fdlr.n, axis=2)
            println(maximum(abs.(real(Σ_freq_compare)-ΣR)))
            Σ_dlr = tau2dlr(fdlr, Σ, axis=2)

            pic = plot(fdlr.ω, real(Σ_dlr)[kF_label,:])
            display(pic)
            readline()
        end
        Σ_shift = ΣR[kF_label,ω0_label]+real(Σ0)[kF_label]
        ΣR = broadcast(+, real(Σ0) .- Σ_shift, ΣR)
        ΣI = broadcast(+, imag(Σ0), ΣI)
        println(Σ_shift)
        if abs((Σ_shift-Σ_shift_old)/Σ_shift)<1e-8
            break
        end
        Σ_shift_old = Σ_shift
    end
    
    Σ_freq = tau2matfreq(fdlr, Σ, adlr.n, axis=2, sumrule = sumrule)
    ΣR = real(Σ_freq)
    ΣI = imag(Σ_freq)
    Σ_shift = ΣR[kF_label,ω0_a_label]+real(Σ0)[kF_label]
    ΣR = broadcast(+, real(Σ0) .- Σ_shift, ΣR)
    ΣI = broadcast(+, imag(Σ0), ΣI)
    println(ΣR[kF_label,:])
    println(ΣI[kF_label,:])
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
        for (ni, n) in enumerate(adlr.n)
            @printf(f, "%32.17g\t%32.17g\n", ΣR[ki,ni], ΣI[ki,ni])
        end
    end


end

function main_GW0_sym(istest=false)
    println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
    println(G0_τ(1.0, EPS), "\t", G0_τ(1.0, -EPS))
    fdlr = DLRGrid( ΣEUV, β, 1e-10, true, :none)
    bdlr = DLRGrid(bEUV, β, 1e-10, false, :ph)
    sdlr = DLRGrid(bEUV, β, 1e-10, true, :ph)
    adlr = DLRGrid(fEUV, β, 1e-10, true, :pha)
    println(fdlr.n)    
    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    # kgrid2 = CompositeGrid(kpanel, order÷2, :cheb)
    # qgrids2 = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order÷2, :gaussian) for k in kgrid.grid] # qgrid for each k
    kF_label = searchsortedfirst(kgrid.grid, kF)
    ω0_label = searchsortedfirst(adlr.n, 0)
    println(fdlr.n[ω0_label])
    ω0_a_label = 1
    println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    kernal_bare, kernal_freq = legendre_dc_sigma(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    kernal_s = real(matfreq2tau(bdlr, kernal_freq, sdlr.τ, axis=3))
    kernal_a = real(matfreq2tau(bdlr, kernal_freq, adlr.τ, axis=3))

    GR ,GI = G0(kgrid, adlr, sdlr)
    Σ0, ΣR, ΣI = calcΣ_sym(GR, GI  , kernal_a, kernal_s, kernal_bare, adlr, sdlr, kgrid, qgrids)
    Σ_shift_old = 0.0
    
    mix = 0.3
    for i in 1:1000
        Σ_shift, GR_new, GI_new = G_τ_sym(Σ0, ΣR, ΣI, adlr, sdlr, kgrid)
        GR = mix*GR_new + (1-mix)*GR
        GI = mix*GI_new + (1-mix)*GI
        Σ0, ΣR, ΣI = calcΣ_sym(GR, GI, kernal_a, kernal_s, kernal_bare, adlr, sdlr, kgrid, qgrids)
 
        if istest
            Σ_tau_compare = matfreq2tau(fdlr, Σ_freq, fdlr.τ, axis=2)
            Σ_freq_compare = tau2matfreq(fdlr, Σ_tau_compare, fdlr.n, axis=2)
            println(maximum(abs.(real(Σ_freq_compare)-ΣR)))
            Σ_dlr = tau2dlr(fdlr, Σ, axis=2)

            pic = plot(fdlr.ω, real(Σ_dlr)[kF_label,:])
            display(pic)
            readline()
        end
        println("Σshift: $Σ_shift")        
        if abs((Σ_shift-Σ_shift_old)/Σ_shift)<1e-8
            break
        end
        Σ_shift_old = Σ_shift
    end

    ΣR =  real(tau2matfreq(adlr, ΣR, adlr.n, axis=2))
    ΣI =  imag(tau2matfreq(sdlr, ΣI, adlr.n, axis=2))
    Σ_shift = ΣR[kF_label,ω0_label]+real(Σ0)[kF_label]
    ΣR = broadcast(+, real(Σ0) .- Σ_shift, ΣR)
    println("$(ΣI[kF_label, 1:10])\n")
    ΣI = broadcast(+, imag(Σ0), ΣI)
    println("$(ΣI[kF_label, 1:10])\n")

    println(ΣR[kF_label,:])
    println(ΣI[kF_label,:])
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
        for (ni, n) in enumerate(adlr.n)
            @printf(f, "%32.17g\t%32.17g\n", ΣR[ki,ni], ΣI[ki,ni])
        end
    end


end

# function plot_kernel_freq(EUV,istest=false)
#     println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")
#     println(G0_τ(1.0, EPS), "\t", G0_τ(1.0, -EPS))
#     fdlr = DLR.DLRGrid(:fermi, EUV, β, 1e-10)
#     adlr = DLR.DLRGrid(:acorr, fEUV, β, 1e-10)
#     bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10) 
#     kpanel = KPanel(Nk, kF, maxK, minK)
#     kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
#     kgrid = CompositeGrid(kpanel, order, :cheb)
#     qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
#     # kgrid2 = CompositeGrid(kpanel, order÷2, :cheb)
#     # qgrids2 = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order÷2, :gaussian) for k in kgrid.grid] # qgrid for each k
#     kF_label = searchsortedfirst(kgrid.grid, kF)
#     qF_label = searchsortedfirst(qgrids[kF_label].grid, kF)
#     ωend_label = searchsortedfirst(bdlr.ωn, 10)
#     ω0_label = 1 #searchsortedfirst(fdlr.n, 0)
#     println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
#     println("kgrid number: $(length(kgrid.grid))")
#     println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

#     ωp = sqrt(8/3/π)*e0
#     kernal_bare, kernal_freq = legendre_dc_sigma(bdlr, kgrid, qgrids, kpanel_bose, order_int)

#     pic = plot(xlabel = L"$\omega_{n}/E_F$", ylabel = L"$W_{l=0}(k_F,k_F,\omega_{n})$", legend=:none)#:bottomright)
#     plot!(pic,bdlr.ωn[1:ωend_label], kernal_freq[kF_label,qF_label,1:ωend_label] .+ kernal_bare[kF_label,qF_label], label = :none, linewidth = 2)
#     plot!(pic,[ωp, ωp], [0.0, kernal_bare[kF_label,qF_label]], label = L"$\omega_{p}$", linestyle=:dot,linewidth = 2)
#     annotate!(pic, ωp+0.5, 120.0, L"$\omega_{p}$", :color)
#     # display(pic)
#     # readline()
#     savefig(pic,"w_l_freq.pdf")
# end

if abspath(PROGRAM_FILE) == @__FILE__
    #check_Σ0(rundir * "/" * ARGS[2])
    #plot_kernel_freq(ΣEUV)

    # fdlr = DLRGrid( ΣEUV, β, 1e-10, true, :none)
    # bdlr = DLRGrid(bEUV, β, 1e-10, false, :ph)
    # sdlr = DLRGrid(bEUV, β, 1e-10, true, :ph)
    # adlr = DLRGrid(fEUV, β, 1e-10, true, :pha)
    # kpanel = KPanel(Nk, kF, maxK, minK)
    # kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    # kgrid = CompositeGrid(kpanel, order, :cheb)
    # qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    # # kgrid2 = CompositeGrid(kpanel, order÷2, :cheb)
    # # qgrids2 = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order÷2, :gaussian) for k in kgrid.grid] # qgrid for each k
    # kF_label = searchsortedfirst(kgrid.grid, kF)
    # ω0_label = searchsortedfirst(fdlr.n, 0)
    # ΣR = zeros(ComplexF64, (length(kgrid.grid), adlr.size))
    # ΣI = zeros(ComplexF64, (length(kgrid.grid), sdlr.size))
    # GR ,GI = G0(kgrid, adlr, sdlr)
    # GR2, GI2 = G_τ_sym(ΣR, ΣI, adlr , sdlr, kgrid)
    # G, GR3, GI3 = G0(kgrid, fdlr)

    

    # GR4 = tau2tau(adlr, GR, fdlr.τ, axis=2)
    # GI4 = tau2tau(sdlr, GI, fdlr.τ, axis=2)
    # #GR4 = tau2matfreq(adlr, GR, fdlr.n, axis=2)
    # #GI4 = tau2matfreq(sdlr, GI, fdlr.n, axis=2)
    # #GR3 = tau2matfreq(fdlr, GR3, fdlr.n, axis=2)
    # #GI3 = tau2matfreq(fdlr, GI3, fdlr.n, axis=2)
    # println(maximum(abs.(GR - GR2)),"," ,maximum( imag(GR)))
    # println(maximum(abs.(GI - GI2)), ",",maximum( imag(GI)))
    # println(maximum(abs.(GR3 - GR4)), ",",maximum( imag(GR4)))
    # println(maximum(abs.(GI3 - GI4)), "," ,maximum( imag(GI4)))
    # println(maximum(abs.(G - GR3 - GI3)), "," , maximum( imag(GR3)))

    # println("kf_label:$(kF_label), $(kgrid.grid[kF_label])")
    # println("kgrid number: $(length(kgrid.grid))")
    # println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    # kernal_bare, kernal_freq = legendre_dc_sigma(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    # kernal = real(matfreq2tau(bdlr, kernal_freq, fdlr.τ, axis=3))
    # kernal_s = real(matfreq2tau(bdlr, kernal_freq, sdlr.τ, axis=3))
    # kernal_a = real(matfreq2tau(bdlr, kernal_freq, adlr.τ, axis=3))


    # Σ0, Σ = calcΣ(kernal, kernal_bare, fdlr, kgrid, qgrids)
    # Σ_freq = tau2matfreq(fdlr, Σ, fdlr.n, axis=2)
    # ΣR = real(Σ_freq)
    # ΣI = imag(Σ_freq)
    # #Σ_shift = ΣR[kF_label,ω0_label]+real(Σ0)[kF_label]
    # #ΣR = broadcast(+, real(Σ0) .- Σ_shift, ΣR)
    # #ΣI = broadcast(+, imag(Σ0), ΣI)

    # ΣR_2, ΣI_2 = calcΣ_sym(GR, GI  , kernal_a, kernal_s, kernal_bare, adlr, sdlr, kgrid, qgrids)

    # ΣR_2 =  real(matfreq2matfreq(adlr, real(ΣR_2), fdlr.n, axis=2))
    # ΣI_2 =  imag(matfreq2matfreq(sdlr, im*ΣI_2, fdlr.n, axis=2))
    # pic1 = plot(fdlr.n, real.(ΣR[kF_label,:]))
    # plot!(pic1, fdlr.n, real.(ΣR_2[kF_label,:]))
    # display(pic1)
    # #savefig(pic1, "kl_escale.pdf")
    # readline()
    # println(maximum(abs.(ΣR - ΣR_2)))
    # println(maximum(abs.(ΣI - ΣI_2)))
    if sigma_type == :gw0
        #main_GW0(false)
        main_GW0_sym(false)
    elseif sigma_type == :g0w0
        # test2 = main_G0W0(10ΣEUV)
        test1 = main_G0W0(ΣEUV)
        # fdlr = DLRGrid(ΣEUV, β, 1e-10, true, :none)
        # fdlr2 = DLRGrid(10ΣEUV, β, 1e-10, true, :none)
        # kpanel = KPanel(Nk, kF, maxK, minK)
        # kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
        # kgrid = CompositeGrid(kpanel, order, :cheb)
        # qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid]
        # test_compare = matfreq2dlr(fdlr2, test2, axis=2)
        # test_compare = dlr2matfreq(fdlr2, test_compare, fdlr.n, axis=2)
        # println("$(maximum(abs.(test_compare - test1)))")
    end

end

