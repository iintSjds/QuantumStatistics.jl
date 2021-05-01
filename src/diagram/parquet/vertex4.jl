struct Para
    chan::Vector{Int}
    # bubble::Dict{Int,Tuple{Vector{Int},Vector{Int}}} 
    # key: channels that are bubbles; 
    # value: (lver, rver) where lver is the lists of allowed channels in the left sub-vertex
    # rver is the lists of allowed channels in the right sub-vertex
    F::Vector{Int}
    V::Vector{Int}
    interactionTauNum::Vector{Int} # list of possible τ degrees of freedom of the bare interaction 0, 2, or 4
    function Para(chan, interactionTauNum)

        for tnum in interactionTauNum
            @assert tnum == 1 || tnum == 2 || tnum == 4
        end

        # chan = [i for i in 1:length(chantype)]
        # for k in keys(bubble) # check validity of the bubble dictionary
        #     @assert issubset(k, chan) "$k isn't in the channel list $chan"
        #     lver, rver = bubble[k]
        #     @assert issubset(lver, chan) "$lver isn't in the channel list $chan"
        #     @assert issubset(rver, chan) "$rver isn't in the channel list $chan"
        # end

        # for (ci, type) in chan
        #     if type == :T

        #     elseif type == :S
        #     elseif type ==:U
        #     else
        #         error("chan $type has not yet been implemented!")
        #     end
        # end
        for c in chan
            @assert c in Allchan "$chan $c isn't implemented!"
        end
        F = intersect(chan, Fchan)
        V = intersect(chan, Vchan)

        return new(chan, F, V, interactionTauNum)
    end
end

struct Green{W}
    Tpair::Vector{Tuple{Int,Int}}
    weight::Vector{W}
    function Green{W}() where {W} 
        return new{W}([], [])
    end
end

# add Tpairs to Green's function (in, out) or vertex4 (inL, outL, inR, outR)
function addTidx(obj, _Tidx)
    for (i, Tidx) in enumerate(obj.Tpair)
        if Tidx == _Tidx
            return i
        end
    end
    push!(obj.Tpair, _Tidx)
    push!(obj.weight, zero(eltype(obj.weight))) # add zero to the weight table of the object
    return length(obj.Tpair)
end

struct IdxMap
    lv::Int # left sub-vertex index
    rv::Int # right sub-vertex index
    G0::Int # shared Green's function index
    Gx::Int # channel specific Green's function index
    ver::Int # composite vertex index
end

struct Bubble{_Ver4,W} # template Bubble to avoid mutually recursive struct
    chan::Int
    Lver::_Ver4
    Rver::_Ver4
    map::Vector{IdxMap}

    function Bubble{_Ver4,W}(ver4::_Ver4, chan::Int, oL::Int, para::Para, level::Int) where {_Ver4,W}
        @assert chan in para.chan "$chan isn't a bubble channels!"
        @assert oL < ver4.loopNum "LVer loopNum must be smaller than the ver4 loopNum"

        oR = ver4.loopNum - 1 - oL # loopNum of the right vertex
        LTidx = ver4.Tidx  # the first τ index of the left vertex
        maxTauNum = maximum(para.interactionTauNum) # maximum tau number for each bare interaction
        RTidx = LTidx + (oL + 1) * maxTauNum + 1  # the first τ index of the right sub-vertex

        if chan == T || chan == U
            LsubVer = para.F
            RsubVer = para.chan
        elseif chan == S
            LsubVer = para.V
            RsubVer = para.chan
        else
            error("chan $chan isn't implemented!")
        end

        Lver = _Ver4{W}(oL, LTidx, para; chan=LsubVer, level=level + 1)
        Rver = _Ver4{W}(oR, RTidx, para; chan=RsubVer, level=level + 1)

        @assert Lver.Tidx == ver4.Tidx "Lver Tidx must be equal to vertex4 Tidx! LoopNum: $(ver4.loopNum), LverLoopNum: $(Lver.loopNum), chan: $chan"

        ############## construct IdxMap ########################################
        map = []
        G = ver4.G
        for (lt, LvT) in enumerate(Lver.Tpair)
            for (rt, RvT) in enumerate(Rver.Tpair)
                GT0 = (LvT[OUTR], RvT[INL])
                GT0idx = addTidx(G[1], GT0)
                GTxidx, VerTidx = 0, 0

                if chan == T
                    VerT = (LvT[INL], LvT[OUTL], RvT[INR], RvT[OUTR])
                    GTx = (RvT[OUTL], LvT[INR])
                elseif chan == U
                    VerT = (LvT[INL], RvT[OUTR], RvT[INR], LvT[OUTL])
                    GTx = (RvT[OUTL], LvT[INR])
                elseif chan == S
                    VerT = (LvT[INL], RvT[OUTL], LvT[INR], RvT[OUTR])
                    GTx = (LvT[OUTL], RvT[INR])
                else
                    throw("This channel is invalid!")
                end

                VerTidx = addTidx(ver4, VerT)
                GTxidx = addTidx(G[chan], GTx)

                for tpair in ver4.Tpair
                    @assert tpair[1] == ver4.Tidx "InL Tidx must be the same for all Tpairs in the vertex4"
                end

                ###### test if the internal + exteranl variables is equal to the total 8 variables of the left and right sub-vertices ############
                Total1 = vcat(collect(LvT), collect(RvT))
                Total2 = vcat(collect(GT0), collect(GTx), collect(VerT))
                @assert compare(Total1, Total2) "chan $(ChanName[chan]): G0=$GT0, Gx=$GTx, external=$VerT don't match with Lver4 $LvT and Rver4 $RvT" 

                push!(map, IdxMap(lt, rt, GT0idx, GTxidx, VerTidx))
            end
        end
        return new(chan, Lver, Rver, map)
    end
end

struct Ver4{W}
    ###### vertex topology information #####################
    level::Int
    
    #######  vertex properties   ###########################
    loopNum::Int
    chan::Vector{Int} # list of channels
    Tidx::Int # inital Tidx

    ######  components of vertex  ##########################
    G::SVector{16,Green}  # large enough to host all Green's function
    bubble::Vector{Bubble{Ver4}}

    ####### weight and tau table of the vertex  ###############
    Tpair::Vector{Tuple{Int,Int,Int,Int}}
    weight::Vector{W}

    function Ver4{W}(loopNum, tidx, para::Para; chan=nothing, level=1) where {W}
        if isnothing(chan)
            chan = para.chan
        end
        g = @SVector [Green{W}() for i = 1:16]
        ver4 = new{W}(level, loopNum, chan, tidx, g, [], [], [])
        @assert loopNum >= 0
        if loopNum == 0
            # bare interaction may have one, two or four independent tau variables
            if 1 in para.interactionTauNum  # instantaneous interaction
                addTidx(ver4, (tidx, tidx, tidx, tidx)) 
            end
            if 2 in para.interactionTauNum  # interaction with incoming and outing τ varibales
                addTidx(ver4, (tidx, tidx, tidx + 1, tidx + 1))  # direct dynamic interaction
                addTidx(ver4, (tidx, tidx + 1, tidx + 1, tidx))  # exchange dynamic interaction
            end
            if 4 in para.interactionTauNum  # interaction with incoming and outing τ varibales
                addTidx(ver4, (tidx, tidx + 1, tidx + 2, tidx + 3))  # direct dynamic interaction
                addTidx(ver4, (tidx, tidx + 3, tidx + 2, tidx + 1))  # exchange dynamic interaction
            end
            return ver4
        end
        for c in para.chan
            for ol = 0:loopNum - 1
                bubble = Bubble{Ver4,W}(ver4, c, ol, para, level)
                if length(bubble.map) > 0  # if zero, bubble diagram doesn't exist
                    push!(ver4.bubble, bubble)
                end
            end
        end
        # TODO: add envolpe diagrams
        # for c in II
        # end
        test(ver4) # more test
        return ver4
    end
end

function compare(A, B)
    # check if the elements of XY are the same as Z
    XY, Z = copy(A), copy(B)
    for e in XY
        if (e in Z) == false
            return false
        end
        Z = (idx = findfirst(x -> x == e, Z)) > 0 ? deleteat!(Z, idx) : Z
    end
    return length(Z) == 0 
end

function test(ver4)
    if length(ver4.bubble) == 0
        return
    end

    G = ver4.G
    for bub in ver4.bubble
        Lver, Rver = bub.Lver, bub.Rver
        for map in bub.map
            LverT, RverT = collect(Lver.Tpair[map.lv]), collect(Rver.Tpair[map.rv]) # 8 τ variables relevant for this bubble
            G1T, GxT = collect(G[1].Tpair[map.G0]), collect(G[bub.chan].Tpair[map.Gx]) # 4 internal variables
            ExtT = collect(ver4.Tpair[map.ver]) # 4 external variables
            @assert compare(vcat(G1T, GxT, ExtT), vcat(LverT, RverT)) "chan $(ChanName[bub.chan]): G1=$G1T, Gx=$GxT, external=$ExtT don't match with Lver4 $LverT and Rver4 $RverT" 
        end
    end
end

function showTree(ver4, para::Para; verbose=0, depth=999)

    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    ete = pyimport("ete3")
    tree = pyimport("tree")

    function tpair(ver4)
        s = ""
        for T in ver4.Tpair
            s *= "($(T[1]), $(T[2]), $(T[3]), $(T[4]))" 
        end
        return s
    end

    function treeview(ver4, t=nothing)
        if isnothing(t)
            t = ete.Tree(name=" ")
        end

        if ver4.loopNum == 0 || ver4.level > depth
            nt = t.add_child(name=tpair(ver4))
            return t
        else
            prefix = "$(ver4.loopNum) lp, $(length(ver4.Tpair)) elem"
            nt = t.add_child(name=prefix * ", ⨁")
            name_face = ete.TextFace(nt.name, fgcolor="black", fsize=10)
            nt.add_face(name_face, column=0, position="branch-top")
        end

        for bub in ver4.bubble
            chantype = ChanName[bub.chan]
            nnt = nt.add_child(name="$(chantype)$(ver4.loopNum)Ⓧ")

            name_face = ete.TextFace(nnt.name, fgcolor="black", fsize=10)
            nnt.add_face(name_face, column=0, position="branch-top")

            treeview(bub.Lver, nnt)
            treeview(bub.Rver, nnt)
        end

        return t
    end

    t = treeview(ver4)
    # style = ete.NodeStyle()
    # style["bgcolor"] = "Khaki"
    # t.set_style(style)


    ts = ete.TreeStyle()
    ts.show_leaf_name = true
    # ts.show_leaf_name = True
    # ts.layout_fn = my_layout
    ####### show tree vertically ############
    # ts.rotation = 90 #show tree vertically

    ####### show tree in an arc  #############
    # ts.mode = "c"
    # ts.arc_start = -180
    # ts.arc_span = 180
    t.show(tree_style=ts)
end

# function eval(ver4::Ver4, KinL, KoutL, KinR, KoutR, Kidx::Int, fast=false)
#     if ver4.loopNum == 0
#         DiagType == POLAR ?
#         ver4.weight[1] = interaction(KinL, KoutL, KinR, KoutR, ver4.inBox, norm(varK[0])) :
#         ver4.weight[1] = interaction(KinL, KoutL, KinR, KoutR, ver4.inBox)
#         return
#     end

#     # LoopNum>=1
#     for w in ver4.weight
#         w .= 0.0 # initialize all weights
#     end
#     G = ver4.G
#     K, Kt, Ku, Ks = (varK[Kidx], ver4.K[1], ver4.K[2], ver4.K[3])
#     eval(G[1], K, varT)
#     bubWeight = counterBubble(K)

#     for c in ver4.chan
#         if c == T || c == TC
#             Kt .= KoutL .+ K .- KinL
#             if (!ver4.inBox)
#                 eval(G[T], Kt)
#             end
#         elseif c == U || c == UC
#             # can not be in box!
#             Ku .= KoutR .+ K .- KinL
#             eval(G[U], Ku)
#         else
#             # S channel, and cann't be in box!
#             Ks .= KinL .+ KinR .- K
#             eval(G[S], Ks)
#         end
#     end
#     for b in ver4.bubble
#         c = b.chan
#         Factor = SymFactor[c] * PhaseFactor
#         Llopidx = Kidx + 1
#         Rlopidx = Kidx + 1 + b.Lver.loopNum

#         if c == T || c == TC
#             eval(b.Lver, KinL, KoutL, Kt, K, Llopidx)
#             eval(b.Rver, K, Kt, KinR, KoutR, Rlopidx)
#         elseif c == U || c == UC
#             eval(b.Lver, KinL, KoutR, Ku, K, Llopidx)
#             eval(b.Rver, K, Ku, KinR, KoutL, Rlopidx)
#         else
#             # S channel
#             eval(b.Lver, KinL, Ks, KinR, K, Llopidx)
#             eval(b.Rver, K, KoutL, Ks, KoutR, Rlopidx)
#         end

#         rN = length(b.Rver.weight)
#         gWeight = 0.0
#         for (l, Lw) in enumerate(b.Lver.weight)
#             for (r, Rw) in enumerate(b.Rver.weight)
#                 map = b.map[(l - 1) * rN + r]

#                     if ver4.inBox || c == TC || c == UC
#                     gWeight = bubWeight * Factor
#                 else
#                     gWeight = G[1].weight[map.G] * G[c].weight[map.Gx] * Factor
#                 end

#                 if fast && ver4.level == 0
#                     pair = ver4.Tpair[map.ver]
#                     dT =
#                         varT[pair[INL]] - varT[pair[OUTL]] + varT[pair[INR]] -
#                         varT[pair[OUTR]]
#                     gWeight *= cos(2.0 * pi / Beta * dT)
#                     w = ver4.weight[ChanMap[c]]
#                 else
#                     w = ver4.weight[map.ver]
#                 end

#                 if c == T || c == TC
#                     w[DI] +=
#                         gWeight *
#                         (Lw[DI] * Rw[DI] * SPIN + Lw[DI] * Rw[EX] + Lw[EX] * Rw[DI])
#                     w[EX] += gWeight * Lw[EX] * Rw[EX]
#                 elseif c == U || c == UC
#                     w[DI] += gWeight * Lw[EX] * Rw[EX]
#                     w[EX] +=
#                         gWeight *
#                         (Lw[DI] * Rw[DI] * SPIN + Lw[DI] * Rw[EX] + Lw[EX] * Rw[DI])
#                 else
#                     # S channel,  see the note "code convention"
#                     w[DI] += gWeight * (Lw[DI] * Rw[EX] + Lw[EX] * Rw[DI])
#                     w[EX] += gWeight * (Lw[DI] * Rw[DI] + Lw[EX] * Rw[EX])
#                 end

#             end
#         end

#     end
# end

# function _expandBubble(children, text, style, bub::Bubble, parent)
#     push!(children, zeros(Int, 0))
#     @assert parent == length(children)
#     dict = Dict(
#         I => ("I", "yellow"),
#         T => ("T", "red"),
#         TC => ("Tc", "pink"),
#         U => ("U", "blue"),
#         UC => ("Uc", "navy"),
#         S => ("S", "green"),
#     )
#     push!(text, "$(dict[bub.chan][1])\n$(bub.Lver.loopNum)-$(bub.Rver.loopNum)")
#     push!(style, "fill:$(dict[bub.chan][2])")

#     current = length(children) + 1
#     push!(children[parent], current)
#     _expandVer4(children, text, style, bub.Lver, current) # left vertex 

#     current = length(children) + 1
#     push!(children[parent], current)
#     _expandVer4(children, text, style, bub.Rver, current) # right vertex
# end

# function _expandVer4(children, text, style, ver4::Ver4, parent)
#     push!(children, zeros(Int, 0))
#     @assert parent == length(children)
#     # println("Ver4: $(ver4.level), Bubble: $(length(ver4.bubble))")
#     if ver4.loopNum > 0
#         info = "O$(ver4.loopNum)\nT[$(length(ver4.Tpair))]\n"
#         for t in ver4.Tpair
#             info *= "[$(t[1]) $(t[2]) $(t[3]) $(t[4])]\n"
#         end
#     else
#         info = "O$(ver4.loopNum)"
#     end
#     push!(text, info)

#     ver4.inBox ? push!(style, "stroke-dasharray:3,2") : push!(style, "")

#     for bub in ver4.bubble
#         # println(bub.chan)
#         current = length(children) + 1
#         push!(children[parent], current)
#         _expandBubble(children, text, style, bub, current)
#     end
# end

# function visualize(ver4::Ver4)
#     children, text, style = (Vector{Vector{Int}}(undef, 0), [], [])
#     _expandVer4(children, text, style, ver4, 1)

#     # text = ["one\n(second line)", "2", "III", "four"]
#     # style = ["", "fill:red", "r:14", "opacity:0.7"]
#     # link_style = ["", "stroke:blue", "", "stroke-width:10px"]
#     tooltip = ["pops", "up", "on", "hover"]
#     t = D3Trees.D3Tree(
#         children,
#         text=text,
#         style=style,
#         tooltip=tooltip,
#         # link_style = link_style,
#         title="Vertex4 Tree",
#         init_expand=2,
#     )

#     D3Trees.inchrome(t)
# end