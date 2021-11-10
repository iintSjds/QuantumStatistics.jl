module DecomposedKernel

export DCKernel

using QuantumStatistics: σx, σy, σz, σ0, FastMath, Utility, TwoPoint#, DLR, Spectral
#import Lehmann
using Lehmann
using LegendrePolynomials
using Printf
using Parameters
using CompositeGrids

srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

include(srcdir*"/interaction.jl")
using .Interaction

include(rundir*"/parameter.jl")
using .parameter

@unpack me, kF, rs, e0, EPS, mom_sep2, mass2, channel, test_KL, bEUV, order_int = parameter.Para()

@inline function kernel_integrand(k, p, n, q, β, W)
    g = e0^2
    legendre_x = (k^2 + p^2 - q^2)/2/k/p
    if(abs(abs(legendre_x)-1)<1e-12)
        legendre_x = sign(legendre_x)*1
    end
    return Pl(legendre_x, channel)*4*π*g/q*W(q, n)
end

@inline function kernel0_integrand(k, p, q)
    g = e0^2
    legendre_x = (k^2 + p^2 - q^2)/2/k/p
    if(abs(abs(legendre_x)-1)<1e-12)
        legendre_x = sign(legendre_x)*1
    end
    @assert -1<=legendre_x<=1 "k=$k,p=$p,q=$q"
    return Pl(legendre_x, channel)*4*π*g/q
end

@inline function DressedW(q, n, β, sigma_type, test_KL, interaction_type)
    if(test_KL == true)
        if interaction_type == :rpa
            return -RPA_mass(q, n, β)
        elseif interaction_type == :ko
            return -KO_mass(q, n, β, sigma_type)
        end
    else
        if interaction_type == :rpa
            return RPA(q, n, β)
        elseif interaction_type == :ko
            return KO(q, n, β, sigma_type)
        end
    end
end

struct DCKernel
    sigma_type::Symbol
    interaction_type::Symbol

    β::Float64
    # Nk::Int
    kF::Float64
    # maxK::Float64
    # minK::Float64
    # order::Int

    bdlr::DLR.DLRGrid
    kgrid::CompositeGrid.Composite
    qgrids::Vector{CompositeGrid.Composite}

    kernel_bare::Array{Float64,2}
    kernel::Array{Float64,3}

    function DCKernel( β, Nk,  kF, maxK, minK, order, sigma_type, interaction_type)
        WW(q, n) = DressedW(q, n, β, sigma_type, test_KL, interaction_type)

        bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10)
        kgrid = CompositeGrid.LogDensedGrid(:cheb, [0.0, maxK], [0.0, kF], Nk, minK, order )
        #println(kgrid.grid)
        qgrids = [CompositeGrid.LogDensedGrid(:gauss, [0.0, maxK], [k, kF], Nk, minK, order) for k in kgrid.grid]
        qgridmax = maximum([qg.size for qg in qgrids])
        #println(qgridmax)

        kernel_bare = zeros(Float64, (length(kgrid.grid), (qgridmax)))
        kernel = zeros(Float64, (length(kgrid.grid), (qgridmax), bdlr.size))

        int_grid_base = CompositeGrid.LogDensedGrid(:uniform, [0.0, 2.1*maxK], [0.0, 2kF], 2Nk, 0.01minK, 2)
        for (ki, k) in enumerate(kgrid.grid)
            for (pi, p) in enumerate(qgrids[ki].grid)
                if abs(k - p) > EPS

                    kmp = abs(k-p)<EPS ? EPS : abs(k-p)
                    kpp = k + p
                    im, ip = floor(int_grid_base, kmp), floor(int_grid_base, kpp)
                    int_panel = Float64[]

                    push!(int_panel, kmp)
                    if im<ip
                        for i in im+1:ip
                            push!(int_panel, int_grid_base[i])
                        end
                    end
                    push!(int_panel, kpp)

                    int_panel = SimpleGrid.Arbitrary{Float64}(int_panel)
                    SubGridType = SimpleGrid.GaussLegendre{Float64}
                    subgrids = subgrids = Vector{SubGridType}([])
                    for i in 1:int_panel.size-1
                        _bound = [int_panel[i],int_panel[i+1]]
                        push!(subgrids, SubGridType(_bound,order_int))
                    end
                    
                    int_grid=CompositeGrid.Composite{Float64,SimpleGrid.Arbitrary{Float64},SubGridType}(int_panel,subgrids)

                    data = [kernel0_integrand(k, p, q) for q in int_grid.grid]
                    kernel_bare[ki, pi] = Interp.integrate1D(data, int_grid)

                    for (ni, n) in enumerate(bdlr.n)
                        data = [kernel_integrand(k, p, n, q, β, WW) for q in int_grid.grid]
                        kernel[ki, pi, ni] = Interp.integrate1D(data, int_grid)
                        @assert isfinite(kernel[ki,pi,ni]) "fail kernel at $ki,$pi,$ni, with $(kernel[ki,pi,ni])"
                    end

                else
                    kernel_bare[ki,pi] = 0
                    for (ni, n) in enumerate(bdlr.n)
                        kernel[ki,pi,ni] = 0
                    end
                end
            end
        end
        
        return new(sigma_type, interaction_type ,β,kF, bdlr, kgrid, qgrids, kernel_bare, kernel)
    end

    function DCKernel(fineKernel::DCKernel, β)
        # extrapolate to kernel of β from fineKernel of lower temperature
        @assert β<fineKernel.β "can only extrapolate from low temp to high temp!"
        sigma_type, interaction_type, kgrid, qgrids = fineKernel.sigma_type, fineKernel.interaction_type, fineKernel.kgrid, fineKernel.qgrids
        kF = fineKernel.kF

        bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10)

        kernel_bare = fineKernel.kernel_bare

        fineKernel_dlr = real(DLR.matfreq2dlr(:corr, fineKernel.kernel, fineKernel.bdlr, axis=3))
        kernel = real(DLR.dlr2matfreq(:corr, fineKernel_dlr, fineKernel.bdlr, bdlr.n ./ (β/fineKernel.β), axis=3))

        return new(sigma_type, interaction_type ,β,kF, bdlr, kgrid, qgrids, kernel_bare, kernel)
    end



end

function save(k::DCKernel, filename::String)
    f = open(filename, "w")
    @printf(f, "%32.17g\t%32.17g\t%32.17g\t%32.17g\n", )



end


end



if abspath(PROGRAM_FILE) == @__FILE__
    using Plots

    DecomposedKernel.Parameters.@unpack kF, β, Nk, maxK, minK, order = DecomposedKernel.parameter.Para()
    # println(DecomposedKernel.RPA(1.0, 1))
    # println(DecomposedKernel.RPA_mass(1.0, 1))
    # println(DecomposedKernel.KO(1.0, 1))
    # println(DecomposedKernel.KO_mass(1.0, 1))
    kernel = (DecomposedKernel.DCKernel( β, Nk,kF,maxK, minK, order, :none, :rpa))
    #kernel_fine = (DecomposedKernel.DCKernel( 1e6, Nk,kF,maxK, minK, order, :none, :rpa))
    #kernel2 = DecomposedKernel.DCKernel(kernel_fine, β)
    kF_label = searchsortedfirst(kernel.kgrid.grid, kernel.kF)
    qF_label = searchsortedfirst(kernel.qgrids[kF_label].grid, kernel.kF)
    #println(kernel_fine.kernel[kF_label,qF_label,:])
    println(kernel.kernel[kF_label,qF_label,:])
    #println(kernel2.kernel[kF_label,qF_label,:])
    #println(maximum(abs.(kernel.kernel-kernel2.kernel)))
    #println(maximum(abs.(kernel.kernel)))
    p = plot(kernel.bdlr.ωn[1:8], kernel.kernel[kF_label,qF_label,1:8])
    display(p)
    readline()

end
