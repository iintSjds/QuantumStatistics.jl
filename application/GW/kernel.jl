module DecomposedKernel

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

@unpack me, kF, rs, e0, β, EPS, mom_sep2, mass2, channel = parameter.Para()

struct DCKernel
    β::Float64
    Nk::Int
    kF::Float64
    maxK::Float64
    minK::Float64
    order::Int

    bdlr::DLR.DLRGrid
    kgrid::CompositeGrid.Composite
    qgrids::Vector{CompositeGrid.Composite}

    function DCKernel(β, Nk,  kF, maxK, minK, order)
        bdlr = DLR.DLRGrid(:corr, bEUV, β, 1e-10)
        kgrid = CompositeGrid.LogDensedGrid(:cheb, [0.0, maxK], [0.0, kF], Nk, minK, order )
        qgrids = [CompositeGrid.LogDensedGrid(:gauss, [0.0, maxK], [k, kF], Nk, minK, order) for k in kgrid.grid]

        return new(β, Nk, kF, maxK, minK, order, bdlr, kgrid, qgrids)
    end

end


end



if abspath(PROGRAM_FILE) == @__FILE__
    println(DecomposedKernel.RPA(1.0, 1))
    println(DecomposedKernel.RPA_mass(1.0, 1))
    println(DecomposedKernel.KO(1.0, 1))
    println(DecomposedKernel.KO_mass(1.0, 1))
end
