"""
    HammersteinWienerTransformation(fu, fu_inv, fy, fy_inv)

Hammerstein-Wiener transformation for linearization in transformed coordinates.

The original (nonlinear) system has:
- Nonlinear input `u_nl` (e.g., current i_dq)
- Nonlinear output `y_nl` (e.g., voltage v_dq)

The HW transformation defines a coordinate change for the LTI system:

```asciiart
           ╭─────────────────────────╮
        ╭──┴─╮   ╭─────╮   ╭──────╮  │
  u_nl ─┤ fu ├─u─┤ LTI ├─y─┤fy_inv├──┴─ y_nl
        ╰────╯   ╰─────╯   ╰──────╯
```

The input nonlinearity `fu` may depend on the nonlinear output `y_nl` (which is
available from the state via `g_inner(x)`):

    fu(u_nl, y_nl) → u       (nonlinear input  → LTI input)
    fu_inv(u, y_nl) → u_nl   (LTI input        → nonlinear input)

The output nonlinearity `fy` must NOT depend on the nonlinear input
(that would create an algebraic loop):

    fy(y) → y_nl             (LTI output       → nonlinear output)
    fy_inv(y_nl) → y         (nonlinear output  → LTI output)
"""
struct HammersteinWienerTransformation{FU,FUinv,FY,FYinv}
    fu::FU
    fu_inv::FUinv
    fy::FY
    fy_inv::FYinv
end

"""
    HammersteinWienerModel

A Hammerstein-Wiener model consisting of:
- `lti`: the linearized [`NetworkDescriptorSystem`](@ref) in LTI coordinates
- `hwt`: the [`HammersteinWienerTransformation`](@ref) used

The `metadata` of `lti` contains the operating point:
- `:x0`    — state vector at the operating point
- `:u0`    — LTI input at the operating point:    `u0 = hwt.fu(u_nl0, y_nl0)`
- `:y0`    — LTI output at the operating point:   `y0 = hwt.fy_inv(y_nl0)`
- `:u_nl0` — nonlinear input at the operating point  (e.g., current i_dq)
- `:y_nl0` — nonlinear output at the operating point (e.g., voltage v_dq)
- `:Tf`    — back-transformation from LTI state δx to original state space
"""
struct HammersteinWienerModel{NDS<:NetworkDescriptorSystem, TF<:HammersteinWienerTransformation}
    lti::NDS
    hwt::TF
    metadata::Dict{Symbol,Any}
end
HammersteinWienerModel(lti, hwt) = HammersteinWienerModel(lti, hwt, Dict{Symbol,Any}())

# Normal form: (u_nl=i_dq, y_nl=v_dq) ↔ (u=δS=[δQ,δP], y=δΘ=[δln|V|, δarg(V)])
# where S = V * conj(I) is the apparent power

function _nf_fu(u_nl, y_nl)
    i_c = Complex(u_nl[1], u_nl[2])
    u_c = Complex(y_nl[1], y_nl[2])
    S = u_c * conj(i_c)  # apparent power S = V * I*
    SVector{2}(imag(S), real(S))  # [Q, P]
end
function _nf_fu_inv(u, y_nl)
    Q, P = u
    u_c = Complex(y_nl[1], y_nl[2])
    ic = conj((P + im*Q) / u_c)
    SVector{2}(real(ic), imag(ic))
end
function _nf_fy(y)
    uc = exp(Complex(y[1], y[2]))
    SVector{2}(real(uc), imag(uc))
end
function _nf_fy_inv(y_nl)
    u_r, u_i = y_nl[1], y_nl[2]
    SVector{2}(1/2*log(u_r^2 + u_i^2), atan(u_i, u_r))
end

"""
    NormalFormTransformation()

Predefined [`HammersteinWienerTransformation`](@ref) for power system bus models.

Maps between dq-frame electrical quantities and power/voltage-angle coordinates:
- nonlinear input  `u_nl = i_dq`    ↔  LTI input  `u = [δQ, δP]`
- nonlinear output `y_nl = v_dq`    ↔  LTI output `y = [δln|V|, δarg(V)]`
"""
NormalFormTransformation = HammersteinWienerTransformation(_nf_fu, _nf_fu_inv, _nf_fy, _nf_fy_inv)

NoTransformation() = HammersteinWienerTransformation(
    (u_nl, y_nl) -> u_nl,  # identity
    (u, y_nl) -> u,        # identity
    y -> y,                # identity
    y_nl -> y_nl           # identity
)

"""
    hammerstein_wiener_linearization(vm::VertexModel, hwt::HammersteinWienerTransformation, x0, p0, u_nl0, y_nl0)
    hammerstein_wiener_linearization(vm::VertexModel, hwt::HammersteinWienerTransformation, state)

Linearize the `VertexModel` using the Hammerstein-Wiener transformation `hwt`.

Returns a [`HammersteinWienerModel`](@ref) whose `lti` field is a
`NetworkDescriptorSystem` in the LTI coordinates:

    M δẋ = A δx + B δu
      δy = C δx

where `u` and `y` are the LTI-coordinate inputs/outputs defined by `hwt`.
The operating point is stored in `lti.metadata`; see [`HammersteinWienerModel`](@ref).

The low-level method takes explicit operating-point vectors `x0` (states), `p0`
(parameters), `u_nl0` (nonlinear inputs), and `y_nl0` (nonlinear outputs).

The `state`-based wrapper extracts these vectors from a state dict (defaulting to
`NetworkDynamics.get_defaults_or_inits_dict(vm)`) and additionally checks that the
operating point satisfies the steady-state condition.
"""
function hammerstein_wiener_linearization(vm::VertexModel, hwt::HammersteinWienerTransformation,
                                          x0, p0, u_nl0, y_nl0)
    # Wrap vertex model into clean non-mutating form:
    #   M ẋ = f_inner(x, u_nl)
    #   y_nl = g_inner(x)
    f_inner = function(x, u_nl)
        dx = zeros(typeof(first(x)*first(u_nl)), dim(vm))
        vm.f(dx, x, u_nl, p0, NaN)
        dx
    end
    g_inner = function(x)
        y = similar(x, length(outsym(vm)))
        if fftype(vm) isa PureStateMap
            vm.g(y, x)
        elseif fftype(vm) isa NoFeedForward
            vm.g(y, x, p0, NaN)
        else
            error("FeedForward output functions are not supported for HW linearization")
        end
        y
    end

    # sanity checks at operating point
    @assert maximum(abs.(f_inner(x0, u_nl0))) < 1e-6 "System not at steady state (f ≠ 0)"
    @assert g_inner(x0) ≈ y_nl0 "g_inner(x0) does not match output defaults"

    # Operating point in LTI coordinates
    u0 = hwt.fu(u_nl0, y_nl0)
    y0 = hwt.fy_inv(y_nl0)
    nu = length(u0)   # LTI input dimension

    # Wrap the system in HW coordinates:
    #   M ẋ = f_hw(x, δu)      where δu = u - u0
    #   y   = g_hw(x)
    f_hw = function(x, δu)
        y_nl = g_inner(x)
        u_nl = hwt.fu_inv(u0 + δu, y_nl)
        f_inner(x, u_nl)
    end
    g_hw = function(x)
        y_nl = g_inner(x)
        hwt.fy_inv(y_nl)
    end

    # Jacobians via ForwardDiff at operating point (δu = 0)
    f_at_x = x   -> f_hw(x,  zeros(nu))
    f_at_u = δu  -> f_hw(x0, δu)

    M = vm.mass_matrix
    A = ForwardDiff.jacobian(f_at_x, x0)
    B = ForwardDiff.jacobian(f_at_u, zeros(nu))
    C = ForwardDiff.jacobian(g_hw,   x0)

    ny = size(C, 1)
    D  = zeros(ny, nu)

    _sym    = collect(NetworkDynamics.sym(vm))
    _insym  = [Symbol("u", NetworkDynamics.subscript(i)) for i in 1:nu]
    _outsym = [Symbol("y", NetworkDynamics.subscript(i)) for i in 1:ny]

    nds = NetworkDescriptorSystem(;
        M, A, B, C, D,
        sym=_sym, insym=_insym, outsym=_outsym,
    )

    hwm_metadata = Dict{Symbol,Any}(
        :x0    => x0,
        :p0    => p0,
        :u0    => Vector(u0),
        :y0    => Vector(y0),
        :u_nl0 => u_nl0,
        :y_nl0 => y_nl0,
        :Tf    => (x, u) -> x,
    )

    HammersteinWienerModel(nds, hwt, hwm_metadata)
end

function hammerstein_wiener_linearization(vm::VertexModel, hwt::HammersteinWienerTransformation,
                                          state=NetworkDynamics.get_defaults_or_inits_dict(vm))
    try
        if init_residual(vm, state) > 1e-8
            @warn "The model does not appear to be at a steady state. That is not expected and might lead to errors!"
        end
    catch e
        @error "Error while trying to check if the model is initialized, did you provide all the necessary defaults and initialized the model?"
        rethrow(e)
    end

    x0    = Float64[state[s] for s in sym(vm)]
    p0    = Float64[state[s] for s in psym(vm)]
    u_nl0 = Float64[state[s] for s in insym(vm)]
    y_nl0 = Float64[state[s] for s in outsym(vm)]

    hwm = hammerstein_wiener_linearization(vm, hwt, x0, p0, u_nl0, y_nl0)
    hwm.metadata[:nl_insym] = collect(insym(vm))
    hwm.metadata[:nl_outsym] = collect(outsym(vm))
    hwm.metadata[:original_vm] = vm
    hwm
end

####
#### HWFunction callable struct  (generalized NormalForm)
####

"""
    HWFunction{DIM,NU,NY,FF,HWT}

Callable struct for Hammerstein-Wiener models in network simulations.
Generalizes `NormalForm{DIM,FF}` to arbitrary input/output nonlinearities.

Parameter vector layout: `[A | B | C | D(if FF) | u0 | y0]`
"""
struct HWFunction{DIM,NU,NY,FF,HWT<:HammersteinWienerTransformation}
    hwt::HWT
    function HWFunction(dim, nu, ny, ff, hwt::HammersteinWienerTransformation)
        new{dim, nu, ny, typeof(ff), typeof(hwt)}(hwt)
    end
end

NetworkDynamics.dim(::HWFunction{DIM}) where {DIM} = DIM
NetworkDynamics.fftype(::HWFunction{DIM,NU,NY,FF}) where {DIM,NU,NY,FF} = FF()
_hw_nu(::HWFunction{DIM,NU}) where {DIM,NU} = NU
_hw_ny(::HWFunction{DIM,NU,NY}) where {DIM,NU,NY} = NY

# Parameter vector ranges
_hw_Arange(hw::HWFunction) = 1:dim(hw)^2
_hw_Brange(hw::HWFunction) = @inbounds (1:dim(hw)*_hw_nu(hw)) .+ _hw_Arange(hw)[end]
_hw_Crange(hw::HWFunction) = @inbounds (1:_hw_ny(hw)*dim(hw)) .+ _hw_Brange(hw)[end]
function _hw_Drange(hw::HWFunction)
    if hasff(hw)
        @inbounds (1:_hw_ny(hw)*_hw_nu(hw)) .+ _hw_Crange(hw)[end]
    else
        @inbounds (1:-1) .+ _hw_Crange(hw)[end]
    end
end
_hw_u0range(hw::HWFunction) = @inbounds (1:_hw_nu(hw)) .+ _hw_Drange(hw)[end]
_hw_y0range(hw::HWFunction) = @inbounds (1:_hw_ny(hw)) .+ _hw_u0range(hw)[end]

NetworkDynamics.pdim(hw::HWFunction) = _hw_y0range(hw)[end]

# Parameter vector views
_hw_Aview(hw::HWFunction{DIM},  vec) where {DIM} = reshape(view(vec, _hw_Arange(hw)), DIM, DIM)
_hw_Bview(hw::HWFunction{DIM,NU},  vec) where {DIM,NU} = reshape(view(vec, _hw_Brange(hw)), DIM, NU)
_hw_Cview(hw::HWFunction{DIM,NU,NY},  vec) where {DIM,NU,NY} = reshape(view(vec, _hw_Crange(hw)), NY, DIM)
function _hw_Dview(hw::HWFunction{DIM,NU,NY}, vec) where {DIM,NU,NY}
    hasff(hw) ? reshape(view(vec, _hw_Drange(hw)), NY, NU) : view(vec, _hw_Drange(hw))
end
_hw_u0view(hw::HWFunction, vec) = view(vec, _hw_u0range(hw))
_hw_y0view(hw::HWFunction, vec) = view(vec, _hw_y0range(hw))

# f for HWFunction without D matrix (NoFeedForward)
function (hw::HWFunction{DIM,NU,NY,<:NoFeedForward})(dx, x, u_nl, p, t) where {DIM,NU,NY}
    # reconstruct LTI output: y = C*x + y0, then y_nl = fy(y)
    C = SMatrix{NY, DIM}(_hw_Cview(hw, p))
    y0 = SVector{NY}(_hw_y0view(hw, p))
    y = muladd(C, x, y0)
    y_nl = hw.hwt.fy(y)

    # apply input nonlinearity: u = fu(u_nl, y_nl), delta_u = u - u0
    u0 = SVector{NU}(_hw_u0view(hw, p))
    u = hw.hwt.fu(u_nl, y_nl)
    delta_u = u - u0

    # state dynamics: dx = A*x + B*delta_u
    A = SMatrix{DIM, DIM}(_hw_Aview(hw, p))
    B = SMatrix{DIM, NU}(_hw_Bview(hw, p))
    _dx = muladd(A, x, B * delta_u)
    dx .= _dx
    nothing
end

# g for HWFunction without D matrix (NoFeedForward)
function (hw::HWFunction{DIM,NU,NY,<:NoFeedForward})(out, x, p, t) where {DIM,NU,NY}
    C = SMatrix{NY, DIM}(_hw_Cview(hw, p))
    y0 = SVector{NY}(_hw_y0view(hw, p))
    y = muladd(C, x, y0)
    y_nl = hw.hwt.fy(y)
    out .= y_nl
    nothing
end

# f for HWFunction with D matrix (FeedForward)
# Extended state: x_full = [x; y_nl_constraint]
function (hw::HWFunction{DIM,NU,NY,<:FeedForward})(dx_full, x_full, u_nl, p, t) where {DIM,NU,NY}
    x = view(x_full, 1:DIM)
    y_nl = SVector{NY}(view(x_full, DIM+1:DIM+NY))

    # apply input nonlinearity
    u0 = SVector{NU}(_hw_u0view(hw, p))
    u = hw.hwt.fu(u_nl, y_nl)
    delta_u = u - u0

    # state dynamics
    A = SMatrix{DIM, DIM}(_hw_Aview(hw, p))
    B = SMatrix{DIM, NU}(_hw_Bview(hw, p))
    _dx = muladd(A, x, B * delta_u)
    view(dx_full, 1:DIM) .= _dx

    # output constraint: y = C*x + D*delta_u + y0, y_nl_lti = fy(y)
    C = SMatrix{NY, DIM}(_hw_Cview(hw, p))
    D = SMatrix{NY, NU}(_hw_Dview(hw, p))
    y0 = SVector{NY}(_hw_y0view(hw, p))
    y = muladd(C, x, muladd(D, delta_u, y0))
    y_nl_lti = hw.hwt.fy(y)

    # constraint residual
    view(dx_full, DIM+1:DIM+NY) .= y_nl .- y_nl_lti
    nothing
end

####
#### VertexModel constructor from HammersteinWienerModel
####

function NetworkDynamics.VertexModel(hwm::HammersteinWienerModel)
    nds = hwm.lti
    hwt = hwm.hwt
    meta = hwm.metadata

    A = nds.A
    B = nds.B
    C = nds.C
    D = nds.D
    M = nds.M

    x0  = meta[:x0]
    p0  = meta[:p0]
    u0  = meta[:u0]
    y0  = meta[:y0]
    Tf  = meta[:Tf]

    nfdim = size(A, 1)
    nu = size(B, 2)
    ny = size(C, 1)

    has_ff = !all(iszero, D)
    ff = has_ff ? FeedForward() : NoFeedForward()

    hwf = HWFunction(nfdim, nu, ny, ff, hwt)

    # Build parameter vector
    pdef = zeros(Float64, pdim(hwf))
    _hw_Aview(hwf, pdef) .= A
    _hw_Bview(hwf, pdef) .= B
    _hw_Cview(hwf, pdef) .= C
    if has_ff
        _hw_Dview(hwf, pdef) .= D
    end
    _hw_u0view(hwf, pdef) .= u0
    _hw_y0view(hwf, pdef) .= y0

    # Symbols
    _sym = [_vecsymbol("δx", i) for i in 1:nfdim]
    if has_ff
        for i in 1:ny
            push!(_sym, Symbol("y_nl", NetworkDynamics.subscript(i)))
        end
    end

    Asym = [_matsymbol("A", i, j) for j in 1:nfdim for i in 1:nfdim]
    Bsym = [_matsymbol("B", i, j) for j in 1:nu for i in 1:nfdim]
    Csym = [_matsymbol("C", i, j) for j in 1:nfdim for i in 1:ny]
    Dsym = has_ff ? [_matsymbol("D", i, j) for j in 1:nu for i in 1:ny] : Symbol[]
    u0sym = [Symbol("u₀", NetworkDynamics.subscript(i)) for i in 1:nu]
    y0sym = [Symbol("y₀", NetworkDynamics.subscript(i)) for i in 1:ny]
    _psym = vcat(Asym, Bsym, Csym, Dsym, u0sym, y0sym)

    # Use original VM's insym/outsym from metadata
    nl_insym = hwm.metadata[:nl_insym]
    nl_outsym = hwm.metadata[:nl_outsym]

    _symdef = [s => (;guess=0.0) for s in _sym]
    _psymdef = _psym .=> pdef
    _insymdef = nl_insym .=> meta[:u_nl0]
    _outsymdef = nl_outsym .=> meta[:y_nl0]

    # Mass matrix
    _M = if has_ff
        @assert isdiag(M)
        Diagonal(vcat(diag(M), zeros(eltype(M), ny)))
    else
        M
    end

    # Output function
    if has_ff
        _g = StateMask(nfdim + 1:nfdim + ny)
        _ff = fftype(_g)
    else
        _g = hwf
        _ff = NoFeedForward()
    end

    # Observable function
    original_vm = hwm.metadata[:original_vm]

    # indices of output symbols that also appear in the original state vector
    nl_outsym_in_sym_idxs = [findfirst(isequal(s), sym(original_vm)) for s in nl_outsym]

    obsf = let nfdim=nfdim, nu=nu, ny=ny,
               hwf=hwf, has_ff=has_ff,
               obsf_orig=original_vm.obsf,
               x0_orig=copy(x0),
               p_orig=copy(p0),
               obsdim=length(original_vm.obssym),
               Tf=Tf,
               nl_outsym_in_sym_idxs=nl_outsym_in_sym_idxs
        (out, δz_full, u_nl_in, p, t) -> begin
            δz = has_ff ? view(δz_full, 1:nfdim) : δz_full
            estim_x_range = 1:length(x0_orig)
            estim_obs_range = (1:obsdim) .+ estim_x_range[end]

            # compute y_nl (output in nonlinear coordinates)
            y_nl = if has_ff
                SVector{ny}(view(δz_full, nfdim+1:nfdim+ny))
            else
                _y_buf = similar(out, ny)
                hwf(_y_buf, δz, p, t)
                SVector{ny}(_y_buf)
            end

            # compute u in LTI coordinates for Tf
            u0_v = SVector{nu}(_hw_u0view(hwf, p))
            u_lti = hwt.fu(u_nl_in, y_nl)
            delta_u = u_lti - u0_v

            # reconstructed original states via Tf
            x_buf = view(out, estim_x_range)
            x_buf .= x0_orig .+ Tf(δz, delta_u)

            # override states that coincide with outputs with the actual y_nl values
            # HACK: this override can vastly improve the obsf performance for reasons not entirely clear to me
            for (j, idx) in enumerate(nl_outsym_in_sym_idxs)
                if !isnothing(idx)
                    x_buf[idx] = y_nl[j]
                end
            end

            # reconstructed original observables
            obs_buf = view(out, estim_obs_range)
            obsf_orig(obs_buf, x_buf, u_nl_in, p_orig, t)
            nothing
        end
    end

    _obssym = vcat(collect(sym(original_vm)), collect(obssym(original_vm)))

    # for any out
    outnames = first.(_outsymdef)
    for i in eachindex(_obssym)
        if _obssym[i] ∈ outnames
            _obssym[i] = Symbol(_obssym[i], "_duplicated")
        end
    end

    vm_lin = VertexModel(;
        f=hwf, g=_g,
        sym=_symdef, psym=_psymdef,
        insym=_insymdef, outsym=_outsymdef,
        ff=_ff,
        mass_matrix=_M,
        obsf=obsf, obssym=_obssym,
    )
    if has_pfmodel(original_vm)
        set_pfmodel!(vm_lin, powerflow_model(original_vm))
    end
    set_metadata!(vm_lin, :hwm, hwm)

    state = NetworkDynamics.get_defaults_or_inits_dict(vm_lin)
    for s in sym(vm_lin)
        state[s] = 0
    end
    if init_residual(vm_lin, state) > 1e-8
        @warn "The HW linearized model does not appear to be at a steady state."
    end
    vm_lin
end

"""
    hw_linearization(vm::VertexModel, hwt::HammersteinWienerTransformation, args...)

Linearize `vm` using the Hammerstein-Wiener transformation `hwt` and return a
`VertexModel` ready for use in network simulations.

This is a convenience wrapper combining [`hammerstein_wiener_linearization`](@ref)
and `VertexModel(::HammersteinWienerModel)`.

All extra arguments are forwarded to `hammerstein_wiener_linearization`.
"""
function hw_linearization(vm::VertexModel, hwt::HammersteinWienerTransformation, args...)
    hwm = hammerstein_wiener_linearization(vm, hwt, args...)
    VertexModel(hwm)
end
