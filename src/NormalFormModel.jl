using LinearAlgebra
using NetworkDynamics: fftype, hasff
using ControlSystems: ControlSystems

struct NormalForm{DIM,FF}
    NormalForm(dim,ff) = new{dim,typeof(ff)}()
end
function NormalForm(lti::NamedTuple)
    dim = size(lti.A, 1)
    ff = !hasproperty(lti, :D) || all(iszero, lti.D) ? NoFeedForward() : FeedForward()
    NormalForm(dim, ff)
end
NetworkDynamics.dim(::NormalForm{DIM,FF}) where {DIM,FF} = DIM
NetworkDynamics.pdim(nf::NormalForm) = Θ0range(nf)[end]
NetworkDynamics.fftype(::NormalForm{DIM,FF}) where {DIM,FF} = FF()

_vecsymbol(x, i) = Symbol(x, NetworkDynamics.subscript(i))
_matsymbol(x, i, j) = Symbol(x, NetworkDynamics.subscript(i),"₋", NetworkDynamics.subscript(j))

Arange(nf::NormalForm) = 1:dim(nf)^2
Brange(nf::NormalForm) = @inbounds (1:2*dim(nf)) .+ Arange(nf)[end]
Crange(nf::NormalForm) = @inbounds (1:dim(nf)*2) .+ Brange(nf)[end]
function Drange(nf::NormalForm)
    if hasff(nf)
        @inbounds (1:4) .+ Crange(nf)[end]
    else
        @inbounds (1:-1) .+ Crange(nf)[end]
    end
end
S0range(nf::NormalForm) = @inbounds (1:2) .+ Drange(nf)[end]
Θ0range(nf::NormalForm) = @inbounds (1:2) .+ S0range(nf)[end]

# dim from vec length would e sqrt(length) - 2
Aview(nf::NormalForm{DIM},  vec) where {DIM} = reshape(view(vec, Arange(nf)), DIM, DIM)
Bview(nf::NormalForm{DIM},  vec) where {DIM} = reshape(view(vec, Brange(nf)), DIM, 2)
Cview(nf::NormalForm{DIM},  vec) where {DIM} = reshape(view(vec, Crange(nf)), 2, DIM)
Dview(nf::NormalForm{DIM},  vec) where {DIM} = hasff(nf) ? reshape(view(vec, Drange(nf)), 2, 2) : view(vec, Drange(nf))
S0view(nf::NormalForm{DIM}, vec) where {DIM} = view(vec, S0range(nf))
Θ0view(nf::NormalForm{DIM}, vec) where {DIM} = view(vec, Θ0range(nf))

# f for nf without D matrix
function (nf::NormalForm{DIM,<:NoFeedForward})(dx, x, isum, p, t) where {DIM}
    # calculate voltage for x
    C = SMatrix{2, DIM}(Cview(nf, p))
    Θ0 = SVector{2}(Θ0view(nf, p))
    Θ = muladd(C, x, Θ0)

    uc = exp(Complex(Θ[1], Θ[2]))
    # get complex current
    ic = Complex(isum[1], isum[2])
    # calculate δS input
    S = conj(ic) * uc
    S0 = SVector{2}(S0view(nf, p))
    δQP = SA[imag(S) - S0[2], real(S) - S0[1]]

    # calculate dx output
    A = SMatrix{DIM, DIM}(Aview(nf, p))
    B = SMatrix{DIM, 2}(Bview(nf, p))
    _dx = muladd(A, x, B * δQP)
    dx .= _dx
    nothing
end
# f for nf with D matrix
function (nf::NormalForm{DIM, <:FeedForward})(dx_full, x_full, isum, p, t) where {DIM}
    # x mai contain outputs if has ff
    x = view(x_full, 1:DIM)
    # for the input, we get the complex current from  x_full
    uc = Complex(x_full[end-1], x_full[end])
    # get complex current
    ic = Complex(isum[1], isum[2])
    # calculate δS input
    S = conj(ic) * uc
    S0 = SVector{2}(S0view(nf, p))
    δQP = SA[imag(S) - S0[2], real(S) - S0[1]]

    # calculate dx output
    A = SMatrix{DIM, DIM}(Aview(nf, p))
    B = SMatrix{DIM, 2}(Bview(nf, p))
    _dx = muladd(A, x, B * δQP)

    view(dx_full, 1:DIM) .= _dx

    # we calculate u again via matrices fro the output constraint
    C = SMatrix{2, DIM}(Cview(nf, p))
    D = SMatrix{2, 2}(Dview(nf, p))
    Θ0 = SVector{2}(Θ0view(nf, p))
    _Θ = muladd(C, x, Θ0)
    Θ = muladd(D, δQP, _Θ)
    uc_lti = exp(Complex(Θ[1], Θ[2]))

    dx_full[end-1] = real(uc) - real(uc_lti)
    dx_full[end]   = imag(uc) - imag(uc_lti)
    nothing
end
# ge for nf without D matrix
function (nf::NormalForm{DIM,<:NoFeedForward})(out, x, p, t) where {DIM}
    # calculate voltage for x
    C = SMatrix{2, DIM}(Cview(nf, p))
    Θ0 = SVector{2}(Θ0view(nf, p))
    Θ = muladd(C, x, Θ0)
    uc = exp(Complex(Θ[1], Θ[2]))
    out[1] = real(uc)
    out[2] = imag(uc)
    nothing
end

function nf_linearization(
    vm::VertexModel,
    state=NetworkDynamics.get_defaults_or_inits_dict(vm);
    transform_constraints=false,
    reduction = 0,
    residualization = nothing,
)
    lti = get_LTI(vm, state)
    if transform_constraints
        lti = reorder_constraints(lti)
        lti = solve_constraints(lti)
    end
    if reduction > 0
        isnothing(residualization) && error("Must specify residualization true/false when specifying order")
        lti = separate_integrator(lti)
        lti = balanced_truncation(lti; reduction, residualization)
    end

    nf = NormalForm(lti)
    nfdim = dim(nf)
    pdef = Float64[-1 for _ in 1:pdim(nf)]
    Aview(nf, pdef) .= lti.A
    Bview(nf, pdef) .= lti.B
    Cview(nf, pdef) .= lti.C
    if hasff(nf)
        Dview(nf, pdef) .= lti.D
    end
    S0view(nf, pdef) .= lti.S0
    Θ0view(nf, pdef) .= lti.Θ0

    _sym = [_vecsymbol("δx", i) for i in 1:nfdim]
    if hasff(nf)
        append!(_sym, [:busbar₊u_r, :busbar₊u_i])
    end
    Asym = [_matsymbol("A", i, j) for j in 1:nfdim for i in 1:nfdim]
    Bsym = [_matsymbol("B", i, j) for j in 1:2 for i in 1:nfdim]
    Csym = [_matsymbol("C", i, j) for j in 1:nfdim for i in 1:2]
    Dsym = if hasff(nf)
        [_matsymbol("D", i, j) for j in 1:2 for i in 1:2]
    else
        Symbol[]
    end
    S0sym = [:S₀_i, :S₀_r]
    Θ0sym = [:Θ₀_r, :Θ₀_i]
    _psym = vcat(Asym, Bsym, Csym, Dsym, S0sym, Θ0sym)
    _insym = [:busbar₊i_r, :busbar₊i_i]
    _outsym = [:busbar₊u_r, :busbar₊u_i]

    # _symdef = [s => (; guess=0.0, init=0.0) for s in _sym]
    _symdef = _sym .=> 0.0
    _psymdef = map(zip(_psym, pdef)) do (sym, def)
        if sym == :Θ₀_i
            sym => (; guess=def, init=def)
        else
            sym => def
        end
    end
    _insymdef = _insym .=> lti.i0
    _outsymdef = _outsym .=> lti.u0

    busbase = PowerDynamics.BusBase(name=:test)
    if length(equations(busbase)) != 6
        @warn "The BusBase model has changed, please check the linearization
               code to add missing busbar observables!"
    end

    # 1-6: P, Q, |u|, angle(u), |i|, angle(i)
    # 7-x: estim x
    # then: estim obs
    obsf = let nfdim=nfdim,
               obsf_orig = vm.obsf,
               x0_orig=copy(lti.x0),
               p_orig=copy(lti.p0),
               obsdim=length(vm.obssym),
               u_r_idx=findfirst(isequal(:busbar₊u_r), sym(vm)),
               u_i_idx=findfirst(isequal(:busbar₊u_i), sym(vm)),
               Tf=lti.Tf
        (out, δz_full, isum, p, t) -> begin
            δz = hasff(nf) ? view(δz_full, 1:nfdim) : δz_full
            busbar_range = 1:6
            estim_x_range = (1:length(x0_orig)) .+ busbar_range[end]
            estim_obs_range = (1:obsdim) .+ estim_x_range[end]

            # the first 6 entries are the busbar observables
            uc = if hasff(nf)
                Complex(δz_full[end-1], δz_full[end])
            else
                uout = view(out, 1:2)
                nf(uout, δz, p, t)
                Complex(uout[1], uout[2])
            end
            ic = Complex(isum[1], isum[2])
            S = -1 * conj(ic) * uc # injector form

            out[1] = real(S)
            out[2] = imag(S)
            out[3] = abs(uc)
            out[4] = angle(uc)
            out[5] = abs(ic)
            out[6] = angle(ic)

            # the next entries are the estimated "original" states
            x_buf = view(out, estim_x_range)
            S0 = S0view(nf, p)
            δQP = SA[imag(S) - S0[2], real(S) - S0[1]]
            x_buf .= x0_orig .+ Tf(δz, δQP)

            # hack: if u_r and u_i explicitly appears in the state vector, we can vastly increase estimation by fillin with actual voltage
            if !isnothing(u_r_idx) && !isnothing(u_i_idx)
                x_buf[u_r_idx] = real(uc)
                x_buf[u_i_idx] = imag(uc)
            end

            # the last states are the estimated "original" observables
            obs_buf = view(out, estim_obs_range)
            obsf_orig(obs_buf, x_buf, isum, p_orig, t)
            nothing # hide
        end
    end
    prefix(s) = Symbol("estim₊", s)
    busbar_obs = [:busbar₊P, :busbar₊Q, :busbar₊u_mag, :busbar₊u_arg, :busbar₊i_mag, :busbar₊i_arg]
    _obssym = vcat(busbar_obs, prefix.(sym(vm)), prefix.(obssym(vm)))

    M = if hasff(nf)
        @assert isdiag(lti.M)
        Diagonal(vcat(diag(lti.M), zeros(eltype(lti.M), 2)))
    else
        lti.M
    end

    # when nf has ff, we handled it to be a pure state map by now
    if hasff(nf)
        _g = StateMask(nfdim + 1:nfdim + 2)
        _ff = fftype(_g)
    else
        _g = nf
        _ff = NoFeedForward()
    end
    vm_lin = VertexModel(;
        f=nf, g=_g,
        sym=_symdef, psym=_psymdef,
        insym=_insymdef, outsym=_outsymdef,
        ff=_ff,
        mass_matrix=M,
        obsf=obsf, obssym=_obssym,
    )
    initf = @initformula :Θ₀_i = atan(:busbar₊u_i, :busbar₊u_r)
    set_initformula!(vm_lin, initf)
    set_pfmodel!(vm_lin, powerflow_model(vm))
    set_metadata!(vm_lin, :lti, lti)

    if init_residual(vm_lin) > 1e-8
        @warn "The linearized model doese not appear to be at a steady state. That is worrisome!"
    end
    vm_lin
end

function reorder_constraints(lti; tol=1e-10)
    (; M, A, B, C, D) = lti

    # SVD of M
    U, s, V = svd(Matrix(M))
    r = count(>(tol), s)   # numerical rank

    # Left transform rescales the first r singular directions
    Σr_inv = Diagonal(1.0 ./ s[1:r])
    Q = [Σr_inv * U[:,1:r]'; U[:,r+1:end]']   # size n×n
    T = V                                     # right transform

    # Transformed system
    _M = Q * M * T
    @assert isdiag(_M)
    _M = Diagonal{Int}(_M)

    _A = Q * A * T
    _B = Q * B
    _C = C * T
    _D = D

    _Tf = (z, u) -> lti.Tf(T * z, u)

    (; M=_M, A=_A, B=_B, C=_C, D=_D, Tf=_Tf,
       S0=lti.S0, Θ0=lti.Θ0, i0=lti.i0, u0=lti.u0, x0=lti.x0, p0=lti.p0)
end

function solve_constraints(lti)
    (; M, A, B, C, D) = lti
    dim = size(A)[1]

    # check that M is in diagonal form and sorted
    @assert isdiag(M) && all(diff(diag(M)) .<= 0) && all(x -> x==0 || x==1, M)

    r = count(==(1), diag(M))   # numerical rank

    A11 = A[1:r, 1:r]
    A12 = A[1:r, r+1:end]
    A21 = A[r+1:end, 1:r]
    A22 = A[r+1:end, r+1:end]
    @assert [A11 A12; A21 A22] == A

    B1 = B[1:r, :]
    B2 = B[r+1:end, :]
    @assert [B1; B2] == B

    C1 = C[:, 1:r]
    C2 = C[:, r+1:end]
    @assert [C1 C2] == C

    # get matrices for reduced model
    M_r = Diagonal(ones(r))
    @assert M[1:r,1:r] == M_r
    A_r = A11 - A12 * (A22 \ A21) |> cleanup_zeros
    B_r = B1 - A12 * (A22 \ B2)   |> cleanup_zeros
    C_r = C1 - C2 * (A22 \ A21)   |> cleanup_zeros
    D_r = D  - C2 * (A22 \ B2)    |> cleanup_zeros

    _Tf = (z, u) -> begin
        z_constraint = -(A22 \ A21)*z - (A22 \ B2)*u
        z_full = vcat(z, z_constraint)
        lti.Tf(z_full, u)
    end

    (; M=M_r, A=A_r, B=B_r, C=C_r, D=D_r, Tf=_Tf,
       S0=lti.S0, Θ0=lti.Θ0, i0=lti.i0, u0=lti.u0, x0=lti.x0, p0=lti.p0)
end

function separate_integrator(lti)
    (; M, A, B, C, D) = lti
    @assert lti.M == Diagonal(ones(size(A)[1])) "Expected mass matrix to be identity"

    # shur decomposition: A = Z*T*Z' , i.e. T = Z'*A*Z is upper triangular
    F = schur(A)
    λs = F.values
    λ0_idxs = findall(λ -> abs(λ) < 1e-10, λs)
    if length(λ0_idxs) == 0
        return lti
    elseif length(λ0_idxs) > 1
        @warn "Separate integrator foudn more than one zero eigenvalue!"
    end
    select = ones(Bool, length(λs))
    select[λ0_idxs] .= false

    Ford = ordschur(F, select)
    Z = Ford.Z
    A_t = Z' * A * Z |> cleanup_zeros
    B_t = Z' * B     |> cleanup_zeros
    C_t = C * Z      |> cleanup_zeros

    _Tf = (z, u) -> begin
        x = Z * z
        lti.Tf(x, u)
    end
    (; A=A_t, B=B_t, C=C_t, Tf=_Tf,
       D=lti.D, M=lti.M, S0=lti.S0, Θ0=lti.Θ0, i0=lti.i0, u0=lti.u0, x0=lti.x0, p0=lti.p0)
end

function balanced_truncation(lti; reduction, residualization)
    (; A, B, C, D) = lti
    Adim = size(A,1)
    @assert lti.M == Diagonal(ones(Adim)) "Mass matrix must be identity for truncation"

    # split system in stable and unstable part by integrator
    integrator_idx = findall(iszero, eachrow(A))
    @assert Set(integrator_idx) == Set(Adim-length(integrator_idx)+1:Adim) "Expected integrator to be in the last rows!"

    srange = 1:Adim-length(integrator_idx)
    mrange = srange[end] + 1:Adim
    Ass = A[srange, srange]
    Asm = A[srange, mrange]
    Bs = B[srange, :]
    Bm = B[mrange, :]
    Cs = C[:, srange]
    Cm = C[:, mrange]

    sdim = length(srange)
    mdim = length(mrange)

    if reduction > sdim
        throw(ArgumentError("The stable subsystem has dimension $sdim, cannot reduce by $(reduction)!"))
    end

    # lets build the matrices for the system to reduce
    reddim = sdim-reduction
    A_tored = Ass
    B_tored = [Bs Asm]
    C_tored = Cs
    D_tored = [D Cm]
    ss = ControlSystems.StateSpace(A_tored, B_tored, C_tored, D_tored)
    ss_bal, _, _ = ControlSystems.baltrunc(ss; residual=residualization, n=reddim)
    # the above only gives reduced σ and T, however we need the full so we balance again
    _, σ, T = ControlSystems.balreal(ss)
    Ar = ss_bal.A
    Br = ss_bal.B
    Cr = ss_bal.C
    Dr = ss_bal.D

    # identification of different matrix subblocks
    indim = size(C,1)
    Br_δS = Br[:, 1:indim]
    Br_δx = Br[:, indim+1:end]
    Dr_δS = Dr[:, 1:indim]
    Dr_δx = Dr[:, indim+1:end]

    # finally build the new system matricies
    Anew = [Ar Br_δx
            zeros(mdim,reddim) zeros(mdim,mdim)]
    Bnew = vcat(Br_δS, Bm)
    Cnew = [Cr Dr_δx]
    Dnew = Dr_δS

    if residualization
        balA = T * A_tored / T
        A21 = balA[reddim+1:end, 1:reddim]
        A22 = balA[reddim+1:end, reddim+1:end]
        B2  = (T*B_tored)[reddim+1:end, :]

        _Tf = (zr_xm, u) -> begin
            # xr_xm = rand(reddim+mdim)
            # u = rand(indim)
            zr = @views zr_xm[1:reddim]
            xm = @views zr_xm[reddim+1:end]
            zr_removed = -(A22 \ A21)*zr - (A22 \ B2)*vcat(u, xm)
            z = vcat(zr, zr_removed)
            xs = T \ z

            # to get the full state back we need to stack  all parts
            xs_xm = vcat(xs, xm)
            lti.Tf(xs_xm, u)
        end
    else
        _Tf = (zr_xm, u) -> begin
            zr = @views zr_xm[1:reddim]
            xm = @views zr_xm[reddim+1:end]
            zr_removed = zeros(reduction)
            z = vcat(zr, zr_removed)
            xs = T \ z
            xs_xm = vcat(xs, xm)
            lti.Tf(xs_xm, u)
        end
    end


    (; A=Anew, B=Bnew, C=Cnew, D=Dnew, Tf=_Tf, M=Diagonal(ones(reddim+mdim)),
        S0=lti.S0, Θ0=lti.Θ0, i0=lti.i0, u0=lti.u0, x0=lti.x0, p0=lti.p0, singularvalues=σ)
end

"""
    cleanup_zeros(M; atol=nothing)

Replace small entries of `M` by 0.

- If `atol` is given, it's used as absolute tolerance.
- If not, a relative tolerance is computed as

    tol = eps(eltype(M)) * norm(M) * size(M,1)

This follows LAPACK-style scaling.
"""
function cleanup_zeros(M; atol=nothing)
    if atol === nothing
        tol = eps(eltype(M)) * norm(M) * size(M,1)
    else
        tol = atol
    end
    map(x -> abs.(x) .< tol ? zero(eltype(M)) : x, M)
end
