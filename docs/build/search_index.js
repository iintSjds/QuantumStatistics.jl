var documenterSearchIndex = {"docs":
[{"location":"lib/green/#Green's-functions-1","page":"Green's functions","title":"Green's functions","text":"","category":"section"},{"location":"lib/green/#","page":"Green's functions","title":"Green's functions","text":"Modules = [QuantumStatistics.Green]","category":"page"},{"location":"lib/green/#QuantumStatistics.Green","page":"Green's functions","title":"QuantumStatistics.Green","text":"Provide N-body response and correlation functions\n\n\n\n\n\n","category":"module"},{"location":"lib/green/#QuantumStatistics.Green.bareFermi-Union{Tuple{T}, Tuple{T,T,AbstractArray{T,1},T}} where T<:AbstractFloat","page":"Green's functions","title":"QuantumStatistics.Green.bareFermi","text":"calcualte with a given momentum vector and the chemical potential μ, rotation symmetry is assumed.\n\n\n\n\n\n","category":"method"},{"location":"lib/green/#QuantumStatistics.Green.bareFermi-Union{Tuple{T}, Tuple{T,T,T}} where T<:AbstractFloat","page":"Green's functions","title":"QuantumStatistics.Green.bareFermi","text":"bareFermi(β, τ, ε, [, scale])\n\nCompute the bare fermionic Green's function. Assume k_B=hbar=1\n\ng(τ0) = e^-ετ(1+e^-βε) g(τ0) = -e^-ετ(1+e^βε)\n\nArguments\n\nβ: the inverse temperature \nτ: the imaginary time, must be (-β, β]\nε: dispersion minus chemical potential: E_k-μ      it could also be the real frequency ω if the bare Green's function is used as the kernel in the Lehmann representation \n\n\n\n\n\n","category":"method"},{"location":"lib/green/#QuantumStatistics.Green.bareFermiMatsubara-Union{Tuple{T}, Tuple{T,Int64,T}} where T<:AbstractFloat","page":"Green's functions","title":"QuantumStatistics.Green.bareFermiMatsubara","text":"bareFermiMatsubara(β, n, ε, [, scale])\n\nCompute the bare Green's function for a given Matsubara frequency.\n\ng(iω_n) = -1(iω_n-ε)\n\nwhere ω_n=(2n+1)πβ. The convention here is consist with the book \"Quantum Many-particle Systems\" by J. Negele and H. Orland, Page 95\n\nArguments\n\nβ: the inverse temperature \nτ: the imaginary time, must be (-β, β]\nε: dispersion minus chemical potential: E_k-μ;       it could also be the real frequency ω if the bare Green's function is used as the kernel in the Lehmann representation \n\n\n\n\n\n","category":"method"},{"location":"lib/green/#QuantumStatistics.Green.FermiDirac-Union{Tuple{T}, Tuple{T,T}} where T<:AbstractFloat","page":"Green's functions","title":"QuantumStatistics.Green.FermiDirac","text":"FermiDirac(β, ε)\n\nCompute the Fermi Dirac function. Assume k_B=hbar=1\n\nf(ϵ) = 1(1+e^-βε)\n\nArguments\n\nβ: the inverse temperature \nε: dispersion minus chemical potential: E_k-μ      it could also be the real frequency ω if the bare Green's function is used as the kernel in the Lehmann representation \n\n\n\n\n\n","category":"method"},{"location":"#QuantumStatistics.jl-1","page":"QuantumStatistics.jl","title":"QuantumStatistics.jl","text":"","category":"section"},{"location":"#","page":"QuantumStatistics.jl","title":"QuantumStatistics.jl","text":"A platform for numerical experiments on quantum statistics.","category":"page"},{"location":"#Outline-1","page":"QuantumStatistics.jl","title":"Outline","text":"","category":"section"},{"location":"#","page":"QuantumStatistics.jl","title":"QuantumStatistics.jl","text":"Pages = [\n    \"lib/green.md\",\n    \"lib/grid.md\",\n    \"lib/fastmath.md\",\n]\nDepth = 1","category":"page"},{"location":"lib/fastmath/#Fast-Math-Functions-1","page":"Fast Math Functions","title":"Fast Math Functions","text":"","category":"section"},{"location":"lib/fastmath/#","page":"Fast Math Functions","title":"Fast Math Functions","text":"Modules = [QuantumStatistics.FastMath]","category":"page"},{"location":"lib/fastmath/#QuantumStatistics.FastMath","page":"Fast Math Functions","title":"QuantumStatistics.FastMath","text":"Provide a set of fast math functions\n\n\n\n\n\n","category":"module"},{"location":"lib/fastmath/#QuantumStatistics.FastMath.invsqrt-Tuple{Float64}","page":"Fast Math Functions","title":"QuantumStatistics.FastMath.invsqrt","text":"invsqrt(x)\n\nThe Legendary Fast Inverse Square Root See the following links: wikipedia and thesis\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#","page":"Fast Math Functions","title":"Fast Math Functions","text":"Modules = [QuantumStatistics.Yeppp]","category":"page"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp","text":"Forked from Yeppp.jl Please go to this link for instructions\n\n\n\n\n\n","category":"module"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.add-Tuple{Array,Array}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.add","text":"add(x::Array, y::Array)\n\nPerform element wise addition of the two array x and y.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.cos!-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.cos!","text":"cos(x)\n\nReturns element wise cos of x.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.cos-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.cos","text":"cos(x)\n\nReturns element wise cos of x.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.dot-Tuple{Array{Float32,1},Array{Float32,1}}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.dot","text":"dot(x::Vector{Float32}, y::Vector{Float32})\n\nCompute the dot product of x and y.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.dot-Tuple{Array{Float64,1},Array{Float64,1}}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.dot","text":"dot(x::Vector{Float64}, y::Vector{Float64})\n\nCompute the dot product of x and y.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.dot-Union{Tuple{T}, Tuple{AbstractArray{T,1},AbstractArray{T,1}}} where T<:AbstractFloat","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.dot","text":"dot(x, y)\n\nCompute the dot product of x and y, which are two vectors with the same dimension.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.evalpoly-Tuple{Any,Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.evalpoly","text":"evalpoly(coeff, x)\n\nEvaluates polynomial with double precision (64-bit) floating-point coefficients coeff on an array of double precision (64-bit) floating-point elements x.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.exp!-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.exp!","text":"exp(x)\n\nComputes the element wise exponential of x inplace.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.exp-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.exp","text":"exp(x)\n\nReturns the element wise exponential of x.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.log!-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.log!","text":"log!(x)\n\nComputes the element wise natural logarithm of x inplace.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.log-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.log","text":"log(x)\n\nReturns the element wise natural logarithm of x.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.max-Tuple{Any,Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.max","text":"max(x, y)\n\nCompares the vectors x and y and return the element wise maxima.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.min-Tuple{Any,Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.min","text":"max(x, y)\n\nCompares the vectors x and y and return the element wise minima.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.multiply-Tuple{Any,Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.multiply","text":"multiply(x, y)\n\nPerform element wise multiplication of the two array x and y.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.sin!-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.sin!","text":"sin(x)\n\nComputes element wise sin of x inplace.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.sin-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.sin","text":"sin(x)\n\nReturns element wise sin of x.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.subtract-Tuple{Array,Array}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.subtract","text":"subtract(x::Array, y::Array)\n\nPerform element wise subtraction of the two array x and y.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.tan!-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.tan!","text":"tan(x)\n\nReturns element wise tan of x.\n\n\n\n\n\n","category":"method"},{"location":"lib/fastmath/#QuantumStatistics.Yeppp.tan-Tuple{Any}","page":"Fast Math Functions","title":"QuantumStatistics.Yeppp.tan","text":"tan(x)\n\nReturns element wise tan of x.\n\n\n\n\n\n","category":"method"},{"location":"lib/grid/#Grids-1","page":"Grids","title":"Grids","text":"","category":"section"},{"location":"lib/grid/#","page":"Grids","title":"Grids","text":"Modules = [QuantumStatistics.Grid]","category":"page"},{"location":"lib/grid/#QuantumStatistics.Grid.Uniform","page":"Grids","title":"QuantumStatistics.Grid.Uniform","text":"Uniform{Type,SIZE}\n\nCreate a uniform Grid with a given type and size\n\nMember:\n\nβ: inverse temperature\nhalfLife: the grid is densest in the range (0, halfLife) and (β-halfLife, β)\nsize: the Grid size\ngrid: vector stores the grid\nsize: size of the grid vector\nhead: grid head\ntail: grid tail\nδ: distance between two grid elements\nisopen: if isopen[1]==true, then grid[1] will be slightly larger than the grid head. Same for the tail.\n\n\n\n\n\n","category":"type"},{"location":"lib/grid/#QuantumStatistics.Grid.boseK","page":"Grids","title":"QuantumStatistics.Grid.boseK","text":"boseK(Kf, maxK, halfLife, size::Int, kFi = floor(Int, 0.5size), twokFi = floor(Int, 2size / 3), type = Float64)\n\nCreate a logarithmic bosonic K Grid, which is densest near the momentum 0 and 2k_F\n\n#Arguments:\n\nKf: Fermi momentum\nmaxK: the upper bound of the grid\nhalfLife: the grid is densest in the range (0, Kf+halfLife) and (2Kf-halfLife, 2Kf+halfLife)\nsize: the Grid size\nkFi: index of Kf\ntwokFi: index of 2Kf\n\n\n\n\n\n","category":"function"},{"location":"lib/grid/#QuantumStatistics.Grid.fermiK","page":"Grids","title":"QuantumStatistics.Grid.fermiK","text":"fermiK(Kf, maxK, halfLife, size::Int, kFi = floor(Int, 0.5size), type = Float64)\n\nCreate a logarithmic fermionic K Grid, which is densest near the Fermi momentum k_F\n\n#Arguments:\n\nKf: Fermi momentum\nmaxK: the upper bound of the grid\nhalfLife: the grid is densest in the range (Kf-halfLife, Kf+halfLife)\nsize: the Grid size\nkFi: index of Kf\n\n\n\n\n\n","category":"function"},{"location":"lib/grid/#QuantumStatistics.Grid.tau","page":"Grids","title":"QuantumStatistics.Grid.tau","text":"tau(β, halfLife, size::Int, type = Float64)\n\nCreate a logarithmic Grid for the imaginary time, which is densest near the 0 and β\n\n#Arguments:\n\nβ: inverse temperature\nhalfLife: the grid is densest in the range (0, halfLife) and (β-halfLife, β)\nsize: the Grid size\n\n\n\n\n\n","category":"function"}]
}
