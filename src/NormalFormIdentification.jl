module NormalFormIdentification

using PowerDynamics
using NetworkDynamics
using PowerDynamics.Library
using LinearAlgebra
using ForwardDiff
using FiniteDiff
using CairoMakie
using Symbolics
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt

export print_equations, get_LTI, print_linearization, bode_plot
export rotational_symmetry

"""
    reduced_jacobian_eigenvalues(system)

Takes a system named tuple and calculates the eigenvalues of the "reduced" jacobian matrix.
"""
function reduced_jacobian_eigenvalues(system)
    A = system.A
    M = system.M
    c_idx = findall(diag(M) .== 0)
    d_idx = findall(diag(M) .== 1)
    f_x = A[d_idx, d_idx] # Differential equations evaluated at the differential variables
    f_y = A[d_idx, c_idx] # Differential equations evaluated at the constrained variables
    g_x = A[c_idx, d_idx] # Constrained equations evaluated at the differential variables
    g_y = A[c_idx, c_idx] # Constrained equations evaluated at the constrained variables
    D = f_y * pinv(g_y) * g_x # Degradation matrix
    jac = f_x - D             # State matrix / Reduced Jacobian (eq. 7.16 in [1])
    eigvals(jac)
end

"""
    bode_plot(G, ωs = 10 .^ range(-2, 2; length=500))

Create a bode plot of the 4 transfer functions of the system `G`.
"""
function bode_plot(G, ωs = 10 .^ range(-2, 2; length=500))
    s_vals = im .* ωs  # s = jω

    fig = Figure(; size=(800, 600))

    output, input = 1, 1
    gains = map(s -> 20 * log10(abs(G(s)[output, input])), s_vals)
    phases = map(s -> angle(G(s)[output, input]) * 180 / pi, s_vals)
    Label(fig[1, 1], L"Bode Plot of $ΔQ \mapsto \mathrm{ln}|V|$", fontsize=16, halign=:center, tellwidth=false)
    ax1 = Axis(fig[2, 1], xlabel="Frequency (rad/s)", ylabel="Gain (dB)", xscale=log10)
    lines!(ax1, ωs, gains, color=:blue, label="Gain")
    ax2 = Axis(fig[3, 1], xlabel="Frequency (rad/s)", ylabel="Phase (deg)", xscale=log10)
    lines!(ax2, ωs, phases, color=:red, label="Phase")

    output, input = 2, 1
    gains = map(s -> 20 * log10(abs(G(s)[output, input])), s_vals)
    phases = map(s -> angle(G(s)[output, input]) * 180 / pi, s_vals)
    Label(fig[4, 1], L"Bode Plot of $ΔQ \mapsto \mathrm{arg}(V)$", fontsize=16, halign=:center, tellwidth=false)
    ax1 = Axis(fig[5, 1], xlabel="Frequency (rad/s)", ylabel="Gain (dB)", xscale=log10)
    lines!(ax1, ωs, gains, color=:blue, label="Gain")
    ax2 = Axis(fig[6, 1], xlabel="Frequency (rad/s)", ylabel="Phase (deg)", xscale=log10)
    lines!(ax2, ωs, phases, color=:red, label="Phase")

    output, input = 1, 2
    gains = map(s -> 20 * log10(abs(G(s)[output, input])), s_vals)
    phases = map(s -> angle(G(s)[output, input]) * 180 / pi, s_vals)

    Label(fig[1, 2], L"Bode Plot of $ΔP \mapsto \mathrm{ln}|V|$", fontsize=16, halign=:center, tellwidth=false)
    ax1 = Axis(fig[2, 2], xlabel="Frequency (rad/s)", ylabel="Gain (dB)", xscale=log10)
    lines!(ax1, ωs, gains, color=:blue, label="Gain")
    ax2 = Axis(fig[3, 2], xlabel="Frequency (rad/s)", ylabel="Phase (deg)", xscale=log10)
    lines!(ax2, ωs, phases, color=:red, label="Phase")

    output, input = 2, 2
    gains = map(s -> 20 * log10(abs(G(s)[output, input])), s_vals)
    phases = map(s -> angle(G(s)[output, input]) * 180 / pi, s_vals)
    Label(fig[4, 2], L"Bode Plot of $ΔP \mapsto \mathrm{arg}(V)$", fontsize=16, halign=:center, tellwidth=false)
    ax1 = Axis(fig[5, 2], xlabel="Frequency (rad/s)", ylabel="Gain (dB)", xscale=log10)
    lines!(ax1, ωs, gains, color=:blue, label="Gain")
    ax2 = Axis(fig[6, 2], xlabel="Frequency (rad/s)", ylabel="Phase (deg)", xscale=log10)
    lines!(ax2, ωs, phases, color=:red, label="Phase")

    fig
end

"""
    get_LTI(vm::VertexModel)

This function retunrs the linearized descriptor system of the given `VertexModel` from
(δQ, δP) => (δθᵢ = δ|V|, δθᵣ = δarg(V)).

It assumes (and checks) that the system is initialized in steady state (otherwise the linearization leads to an affine system).

    M ẋ = A δx + B δQP
      y = C δx         = [δ|V|, δarg(V)]

Notice that this only gives the **change** of the complex phase, not the absolute value.
"""
function get_LTI(vm::VertexModel)
    pvec = NetworkDynamics.get_default_or_init.(Ref(vm), psym(vm))

    # first we wrap the inner functions of the vertex model to get to nice
    # M ẋ = f_inner(x, u)
    #   y = g_inner(x)
    # version without the unnecessary complexity of mutating output and other parmeters
    f_inner = function(x, idq)
        dx = zeros(typeof(first(x)*first(idq)), dim(vm))
        vm.f(dx, x, idq, pvec, NaN)
        dx
    end
    g_inner = function(x)
        vdq = similar(x, 2)
        if fftype(vm) isa PureStateMap
            vm.g(vdq, x)
        elseif fftype(vm) isa NoFeedForward
            vm.g(vdq, x, pvec, NaN)
        else
            error()
        end
        vdq
    end

    # sanity cecks, g_inner should return voltage at eq point, f_inner should return 0
    xvec = Float64.(NetworkDynamics.get_default_or_init.(Ref(vm), sym(vm)))
    idqvec = Float64.(NetworkDynamics.get_default_or_init.(Ref(vm), insym(vm)))
    @assert maximum(abs.(f_inner(xvec, idqvec) - zeros(dim(vm)))) < 1e-6
    @assert g_inner(xvec) ≈ NetworkDynamics.get_default_or_init.(Ref(vm), outsym(vm))

    # for liniearization, we define 1 \mathrm{arg} functions of f around QP0 and f around x0
    S0 = Complex(g_inner(xvec)...) * conj(Complex(idqvec...))
    QP0vec = [imag(S0), real(S0)]

    # next we wrap the f_inner and f_out to get to the desired inputs i.e. ΔQP -> (|V|, arg(V))
    # M ẋ = f(x, ΔQP)
    #   θ = g(x)
    f = function(x, ΔQP)
        Q, P = QP0vec + ΔQP
        u_r, u_i = g_inner(x)
        # NOTE: the sign of the current is not totally clear
        ic = conj( (P+im*Q) / (u_r+im*u_i) )
        idq = [real(ic), imag(ic)]
        f_inner(x, idq)
    end
    g = function(x)
        u_r, u_i = g_inner(x)
        [1/2*log(u_r^2 + u_i^2), atan(u_i, u_r)]
    end

    f_around_ΔQP = x -> f(x, zeros(2))
    f_around_x0  = ΔQP -> f(xvec, ΔQP)

    # The ABC matrices are the jacobians (note that this linearization goes from
    # QP -> θ to δQP -> δθ, so input and output def changes)
    M = vm.mass_matrix
    A = ForwardDiff.jacobian(f_around_ΔQP, xvec) # ∂f/∂x(x0, ΔQP0)
    B = ForwardDiff.jacobian(f_around_x0, zeros(2)) # ∂f/∂ΔQP(x0, ΔQP0)
    C = ForwardDiff.jacobian(g, xvec)      # ∂g/∂x(x0)

    @assert A ≈ FiniteDiff.finite_difference_jacobian(f_around_ΔQP, xvec)
    @assert B ≈ FiniteDiff.finite_difference_jacobian(f_around_x0, zeros(2))
    @assert C ≈ FiniteDiff.finite_difference_jacobian(g, xvec)

    # lastly we define the transfer matrix as a function of s
    # NOTE: during discussion, we sometimes had -1 in the G function
    G = s -> C * inv(s*M - A) * B
    Gs = s -> s * C * inv(s*M - A) * B

    (; M, A, B, C, G, Gs)
end

function rotational_symmetry(vm::VertexModel; covariant=[])
    _vm = copy(vm) # create a copy first
    res = init_residual(_vm)
    if isnan(res) || res > 1e-6
        error("The system is not initialized in steady state, cannot perform rotational symmetry identification.")
    end

    residuals = Float64[]
    for αd in 1:360
        local α = rad2deg(αd)
        D = [cos(α) -sin(α); sin(α) cos(α)]

        # input / output transformation
        u = [get_initial_state(vm, :busbar₊u_r), get_initial_state(vm, :busbar₊u_i)]
        i = [get_initial_state(vm, :busbar₊i_r), get_initial_state(vm, :busbar₊i_i)]
        unew = D * u
        inew = D * i
        set_default!(_vm, :busbar₊u_r, unew[1])
        set_default!(_vm, :busbar₊u_i, unew[2])
        set_default!(_vm, :busbar₊i_r, inew[1])
        set_default!(_vm, :busbar₊i_i, inew[2])

        # covariatn variable transformation
        for sym in covariant
            if sym isa NTuple{2,Symbol}
                phasorlike = [get_initial_state(vm, sym[1]), get_initial_state(vm, sym[2])]
                phasorlike_new = D * phasorlike
                set_default!(_vm, sym[1], phasorlike_new[1])
                set_default!(_vm, sym[2], phasorlike_new[2])
            elseif sym isa Symbol
                δlike = get_initial_state(vm, sym)
                δlike_new = δlike + α
                set_default!(_vm, sym, δlike_new)
            else
                error()
            end
        end

        res = init_residual(_vm)
        push!(residuals, res)
        # printstyled("α = $(lpad(αd, 3))°: ", color=:blue)
        # printstyled(repr(res)*"\n")
    end

    printstyled("Performed rotational symmetry analysis $(length(residuals)) angles.\n"; color=:blue)
    println(" - input current and output voltage was rotated and the residual of the steady state was calculated for each angle.")
    cov_single = filter(s -> s isa Symbol, covariant)
    cov_pair   = filter(s -> s isa NTuple{2,Symbol}, covariant)
    if !isempty(cov_single)
        println(" - assume covariant variables $(join(cov_single, ", ")) transform like δ → δ + Δδ")
    end
    if !isempty(cov_pair)
        println(" - assume covariant variables pairs $(join(cov_pair, ", ")) transform like (u_r, u_i) → D(Δδ).(u_r, u_i)")
    end

    extr = extrema(residuals)
    TOL = 1e-6
    if maximum(abs.(extr)) < TOL
        printstyled("✓ All residuals are below $TOL"; color=:green)
    else
        printstyled("× Some residuals are above $TOL: extrema $(extr)"; color=:red)
    end
end

function print_equations(vm::VertexModel; remove_ns::AbstractVector=[], original_model=nothing)
    if !isnothing(original_model)
        def_eqs = full_equations(original_model)
    end

    subs = Dict(map(eq -> eq.lhs => eq.rhs, vm.metadata[:observed]))
    feqs = ModelingToolkit.fixpoint_sub(vm.metadata[:equations], subs)
    geqs = ModelingToolkit.fixpoint_sub(vm.metadata[:outputeqs], subs)

    if !isnothing(original_model)
        def_str = join(repr.(def_eqs), "\n")
    end
    f_str = join(repr.(feqs), "\n")
    g_str = join(repr.(geqs), "\n")

    remove_ns = string.(remove_ns)
    replacements = [
        "Differential(t)" => "Dt",
        "(t)" => "",
        (ns*"₊" => "" for ns in remove_ns)...
    ]

    for r in replacements
        if !isnothing(original_model)
            def_str = replace(def_str, r)
        end
        f_str = replace(f_str, r)
        g_str = replace(g_str, r)
    end
    if !isnothing(original_model)
        printstyled("defining equations:\n"; bold=true, color=:blue)
        println(def_str)
        println()
    end
    printstyled("f equations:\n"; bold=true, color=:blue)
    println(f_str)
    println()
    printstyled("g equations:\n"; bold=true, color=:blue)
    println(g_str)
end

function print_linearization(vm)
    lti = get_LTI(vm)
    printstyled("Linearzation Point\n"; bold=true, color=:blue)
    dump_initial_state(vm; obs=false)
    # printstyled("x-vector:\n")
    # for s in vm.sym
    #     println("  ", s, " = ", NetworkDynamics.str_significant(get_initial_state(vm, s), sigdigits=5))
    # end
    # printstyled("p-vector:\n")
    # for s in vm.psym
    #     println("  ", s, " = ", NetworkDynamics.str_significant(get_initial_state(vm, s), sigdigits=5))
    # end
    printstyled("M:\n"; bold=true, color=:blue)
    show(stdout, MIME"text/plain"(), lti.M)
    printstyled("\nA:\n"; bold=true, color=:blue)
    show(stdout, MIME"text/plain"(), lti.A)
    printstyled("\nB:\n"; bold=true, color=:blue)
    show(stdout, MIME"text/plain"(), lti.B)
    printstyled("\nC:\n"; bold=true, color=:blue)
    show(stdout, MIME"text/plain"(), lti.C)
    printstyled("\n\nEigenvalue Analysis\n"; bold=true, color=:blue)
    λ = eigvals(lti.A)
    for x in λ
        println("  ", x)
    end
    if any(x -> isapprox(x, 0; atol=1e-10), λ)
        printstyled("  ✓ A has zero mode", color=:green)
    else
        printstyled("  × A has no zero mode", color=:red)
    end

    if lti.M != I
        printstyled("\nReduced Eigenvalues\n"; bold=true, color=:blue)
        λred = reduced_jacobian_eigenvalues(lti)
        for x in λred
            println("  ", x)
        end
        if any(x -> isapprox(x, 0; atol=1e-10), λred)
            printstyled("  ✓ A reduced has zero mode", color=:green)
        else
            printstyled("  × A reduced has no zero mode", color=:red)
        end
    end
    nothing
end

end # module NormalFormIdentification
