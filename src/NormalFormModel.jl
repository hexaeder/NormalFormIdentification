_vecsymbol(x, i) = Symbol(x, NetworkDynamics.subscript(i))
_matsymbol(x, i, j) = Symbol(x, NetworkDynamics.subscript(i),"₋", NetworkDynamics.subscript(j))

_shift(r::UnitRange, offset) = (offset + r.start):(offset + r.stop)
Arange(dim) = 1:dim^2
Brange(dim) = @inbounds _shift(1:2*dim, Arange(dim)[end])
Crange(dim) = @inbounds _shift(1:dim*2, Brange(dim)[end])
S0range(dim) = @inbounds _shift(1:2, Crange(dim)[end])
Θ0range(dim) = @inbounds _shift(1:2, S0range(dim)[end])
nf_pdim(dim) = Θ0range(dim)[end]

Aview(vec, dim) = reshape(view(vec, Arange(dim)), dim, dim)
Bview(vec, dim) = reshape(view(vec, Brange(dim)), dim, 2)
Cview(vec, dim) = reshape(view(vec, Crange(dim)), 2, dim)
S0view(vec, dim) = view(vec, S0range(dim))
Θ0view(vec, dim) = view(vec, Θ0range(dim))

struct NormalForm{DIM}
    NormalForm(dim) = new{dim}()
end
function (::NormalForm{DIM})(dx, x, isum, p, t) where {DIM}
    # calculate voltage for x
    C = SMatrix{2, DIM}(Cview(p, DIM))
    Θ0 = SVector{2}(Θ0view(p, DIM))
    Θ = muladd(C, x, Θ0)
    uc = exp(Complex(Θ[1], Θ[2]))
    # get complex current
    ic = Complex(isum[1], isum[2])
    # calculate δS input
    S = conj(ic) * uc
    S0 = SVector{2}(S0view(p, DIM))
    δQP = SA[imag(S) - S0[2], real(S) - S0[1]]

    # calculate dx output
    A = SMatrix{DIM, DIM}(Aview(p, DIM))
    B = SMatrix{DIM, 2}(Bview(p, DIM))
    _dx = muladd(A, x, B * δQP)
    dx .= _dx
    nothing
end
function (::NormalForm{DIM})(out, x, p, t) where {DIM}
    # calculate voltage for x
    C = SMatrix{2, DIM}(Cview(p, DIM))
    Θ0 = SVector{2}(Θ0view(p, DIM))
    Θ = muladd(C, x, Θ0)
    uc = exp(Complex(Θ[1], Θ[2]))
    out[1] = real(uc)
    out[2] = imag(uc)
    nothing
end

struct NormalFormObsF{DIM}
    NormalFormObsF(dim) = new{dim}()
end
function (::NormalFormObsF{DIM})(out, x, isum, p, t) where {DIM}
    # busbar voltage (temp write in out buf)
    uout = view(out, 1:2)
    NormalForm(DIM)(uout, x, p, t)
    uc = Complex(uout[1], uout[2])
    ic = Complex(isum[1], isum[2])
    S = -1 * conj(ic) * uc # injector form

    out[1] = real(S)
    out[2] = imag(S)
    out[3] = abs(uc)
    out[4] = angle(uc)
    out[5] = abs(ic)
    out[6] = angle(ic)
    nothing # hide
end

function nf_linearization(vm::VertexModel, state=NetworkDynamics.get_defaults_or_inits_dict(vm))
    lti = get_LTI(vm, state)
    xdim = size(lti.A)[1]
    pdef = Float64[-1 for _ in 1:nf_pdim(xdim)]
    Aview(pdef, xdim) .= lti.A
    Bview(pdef, xdim) .= lti.B
    Cview(pdef, xdim) .= lti.C
    S0view(pdef, xdim) .= lti.S0
    Θ0view(pdef, xdim) .= lti.Θ0

    _sym = [_vecsymbol("δx", i) for i in 1:xdim]
    Asym = [_matsymbol("A", i, j) for j in 1:xdim for i in 1:xdim]
    Bsym = [_matsymbol("B", i, j) for j in 1:2 for i in 1:xdim]
    Csym = [_matsymbol("C", i, j) for j in 1:xdim for i in 1:2]
    S0sym = [:S₀_i, :S₀_r]
    Θ0sym = [:Θ₀_r, :Θ₀_i]
    _psym = vcat(Asym, Bsym, Csym, S0sym, Θ0sym)
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
    _obssym = [:busbar₊P, :busbar₊Q, :busbar₊u_mag, :busbar₊u_arg, :busbar₊i_mag, :busbar₊i_arg]

    vm_lin = VertexModel(;
        f=NormalForm(xdim), g=NormalForm(xdim),
        sym=_symdef, psym=_psymdef,
        insym=_insymdef, outsym=_outsymdef,
        ff=NoFeedForward(),
        mass_matrix=lti.M,
        obsf=NormalFormObsF(xdim), obssym=_obssym,
    )
    initf = @initformula :Θ₀_i = atan(:busbar₊u_i, :busbar₊u_r)
    set_initformula!(vm_lin, initf)

    if init_residual(vm_lin) > 1e-8
        @warn "The linearized model doese not appear to be at a steady state. That is worrisome!"
    end
    vm_lin
end
