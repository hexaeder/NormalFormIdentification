using LinearAlgebra
using NetworkDynamics: fftype, hasff

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

_shift(r::UnitRange, offset) = (offset + r.start):(offset + r.stop)
Arange(nf::NormalForm) = 1:dim(nf)^2
Brange(nf::NormalForm) = @inbounds _shift(1:2*dim(nf), Arange(nf)[end])
Crange(nf::NormalForm) = @inbounds _shift(1:dim(nf)*2, Brange(nf)[end])
function Drange(nf::NormalForm)
    if hasff(nf)
        @inbounds _shift(1:4, Crange(nf)[end])
    else
        @inbounds _shift(1:-1, Crange(nf)[end])
    end
end
S0range(nf::NormalForm) = @inbounds _shift(1:2, Drange(nf)[end])
Θ0range(nf::NormalForm) = @inbounds _shift(1:2, S0range(nf)[end])

# dim from vec length would e sqrt(length) - 2
Aview(nf::NormalForm{DIM},  vec) where {DIM} = reshape(view(vec, Arange(nf)), DIM, DIM)
Bview(nf::NormalForm{DIM},  vec) where {DIM} = reshape(view(vec, Brange(nf)), DIM, 2)
Cview(nf::NormalForm{DIM},  vec) where {DIM} = reshape(view(vec, Crange(nf)), 2, DIM)
Dview(nf::NormalForm{DIM},  vec) where {DIM} = hasff(nf) ? reshape(view(vec, Drange(nf)), 2, 2) : view(vec, Drange(nf))
S0view(nf::NormalForm{DIM}, vec) where {DIM} = view(vec, S0range(nf))
Θ0view(nf::NormalForm{DIM}, vec) where {DIM} = view(vec, Θ0range(nf))

function (nf::NormalForm{DIM})(dx_full, x_full, isum, p, t) where {DIM}
    # x mai contain outputs if has ff
    x = hasff(nf) ? view(x_full, 1:DIM) : x_full

    # calculate voltage for x
    C = SMatrix{2, DIM}(Cview(nf, p))
    Θ0 = SVector{2}(Θ0view(nf, p))
    Θ = muladd(C, x, Θ0)

    if hasff(nf)
        D = SMatrix{2, 2}(Dview(nf, p))
        Θ = muladd(D, isum, Θ)
    end

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
    if hasff(nf)
        view(dx_full, 1:DIM) .= _dx
        dx_full[end-1] = x_full[end-1] - real(uc)
        dx_full[end]   = x_full[end]   - imag(uc)
    else
        dx_full .= _dx
    end
    nothing
end
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
function (::NormalForm{DIM,<:FeedForward})(out, x_full) where {DIM}
    out .= x_full[end-1:end]
    nothing
end

function nf_linearization(vm::VertexModel, state=NetworkDynamics.get_defaults_or_inits_dict(vm))
    lti = get_LTI(vm, state)
    lti = reorder_constraints(lti)
    # lti = solve_constraints(lti)

    nf = NormalForm(lti)
    xdim = dim(nf)
    pdef = Float64[-1 for _ in 1:pdim(nf)]
    Aview(nf, pdef) .= lti.A
    Bview(nf, pdef) .= lti.B
    Cview(nf, pdef) .= lti.C
    if hasff(nf)
        Dview(nf, pdef) .= lti.D
    end
    S0view(nf, pdef) .= lti.S0
    Θ0view(nf, pdef) .= lti.Θ0

    _sym = [_vecsymbol("δx", i) for i in 1:xdim]
    if hasff(nf)
        append!(_sym, [:busbar₊u_r, :busbar₊u_i])
    end
    Asym = [_matsymbol("A", i, j) for j in 1:xdim for i in 1:xdim]
    Bsym = [_matsymbol("B", i, j) for j in 1:2 for i in 1:xdim]
    Csym = [_matsymbol("C", i, j) for j in 1:xdim for i in 1:2]
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
    obsf = let xdim=xdim,
               obsf_orig = vm.obsf,
               x0_orig=copy(lti.x0),
               p_orig=copy(lti.p0),
               obsdim=length(vm.obssym),
               u_r_idx=findfirst(isequal(:busbar₊u_r), sym(vm)),
               u_i_idx=findfirst(isequal(:busbar₊u_i), sym(vm)),
               Tf=lti.Tf
        (out, δz_full, isum, p, t) -> begin
            δz = hasff(nf) ? view(δz_full, 1:xdim) : δz_full
            busbar_range = 1:6
            estim_x_range = (1:xdim) .+ busbar_range[end]
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

    vm_lin = VertexModel(;
        f=nf, g=nf,
        sym=_symdef, psym=_psymdef,
        insym=_insymdef, outsym=_outsymdef,
        ff=fftype(nf),
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
    A_r = A11 - A12*inv(A22)*A21
    B_r = B1 - A12*inv(A22)*B2
    C_r = C1 - C2*inv(A22)*A21
    D_r = D - C2*inv(A22)*B2

    # TODO: continue here!
    # i don;t know yet how to get T and Q to go from z to x (old)
    # is this even possible? because i think it also depends on u

    _Tf = (z, u) -> begin
        z_constraint = -inv(A22)*A21*z -inv(A22)*B2*u
        vcat(z, z_constraint)
    end

    (; M=M_r, A=A_r, B=B_r, C=C_r, D=D_r, Tf=_Tf,
       S0=lti.S0, Θ0=lti.Θ0, i0=lti.i0, u0=lti.u0, x0=lti.x0, p0=lti.p0)

end
