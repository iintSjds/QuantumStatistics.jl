# We work with Rydberg units, length scale Bohr radius a_0, energy scale: Ry
module parameter

#using StaticArrays, QuantumStatistics
using Parameters

@with_kw struct Para
    sigma_type = :none
    interaction_type = :rpa
    test_KL::Bool = false
    WID::Int = 1
    me::Float64 = 0.5  # electron mass
    dim::Int = 3    # dimension (D=2 or 3, doesn't work for other D!!!)
    spin::Int = 2  # number of spins
    EPS::Float64 = 1e-11

    rs::Float64 = 3.0
    e0::Float64 = sqrt(rs*2.0/(9π/4.0)^(1.0/3))  #sqrt(2) electric charge
    kF::Float64 = 1.0  #(dim == 3) ? (9π / (2spin))^(1 / 3) / rs : sqrt(4 / spin) / rs
    EF::Float64 = 1.0     #kF^2 / (2me)
    β::Float64 = 3000 / kF^2

    fEUV::Float64 = 100EF
    bEUV::Float64 = 100EF
    ΣEUV::Float64 = 100EF



    mass2::Float64 =  0.01
    mom_sep::Float64 = 0.1
    mom_sep2::Float64 = 1.0
    freq_sep::Float64 = 0.01
    channel::Int = 0

    ### grid  constants ###
    Nk::Int = 8+Int64(floor(log10(β)))
    order::Int = 8
    order_int::Int = 16
    maxK::Float64 = 10.0 * kF
    minK::Float64 =  0.0001/ (β * kF)

end

end
