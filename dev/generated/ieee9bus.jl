# # Linearization of Models in IEEE 9-Bus System
#
# In this example, we'll substitute dynamic models with their
# Normal Form (NF) linearization and validate that the linearized models remain accurate
# under global phase shifts.
#
# In PowerDynamics.jl, we start with a nonlinear dynamical system of the form
#
# ```math
# \begin{aligned}
#   \frac{d\mathbf{x}}{dt} &= f\left(\mathbf{x}, \mathbf{i}\right)\\
#   \mathbf{u} &= g(\mathbf{x})
# \end{aligned}
# ```
# where $\mathbf{u}$ is the complex voltage output and $\mathbf{i}$ is the complex current input.
#
# However, for linearization it's more convenient to work with
# power-phase variables rather than voltage-current variables. To transform this into the
# normal form convention $\mathbf{\Delta S} \mapsto \mathbf{\Delta \Theta}$,
# we need to reformulate the system as
# ```math
# \begin{aligned}
#   \frac{d\mathbf{x}}{dt} &= f\left(\mathbf{x}, \left(\frac{\mathbf{S}}{g(\mathbf{x})}\right)^*\right) &&= \bar{f}(\mathbf{x}, \mathbf{S})\\
#   \mathbf{\Theta} &= \mathrm{ln}\,g(\mathbf{x}) &&= \bar{g}(\mathbf{x})
# \end{aligned}
# ```
#
# Next, we identify the equilibrium point around which we want to linearize the system:
#
# ```math
# \begin{aligned}
#   \frac{d\mathbf{x}}{dt} &= 0 = f(\mathbf{x}_0, \mathbf{i}_0)\\
#     \mathbf{u}_0 &= g(\mathbf{x}_0)\\
#   \mathbf{S}_0 &= \mathbf{u}_0 \cdot \mathbf{i}_0^*\\
#   \mathbf{\Theta}_0 &= \mathrm{ln}\,g(\mathbf{x}_0) = \mathrm{ln}\,\mathbf{u}_0
# \end{aligned}
# ```
# The linearization around the equilibrium point looks liek this:
# ```math
# \begin{aligned}
#   \frac{d\mathbf{x}}{dt} &= \underbrace{\bar{f}(\mathbf{x}_0, \mathbf{S}_0)}_{=0} + \frac{\partial \bar{f}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_0, \mathbf{S}_0}\delta \mathbf{x} +
#   \frac{\partial \bar{f}}{\partial \mathbf{S}}\bigg|_{\mathbf{x}_0, \mathbf{S}_0}\delta\mathbf{S}\\
#   \mathbf{\Theta} &= \mathbf{\Theta}_0 + \frac{\partial \bar{g}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_0} \delta \mathbf{x}
# \end{aligned}
# ```
# where the partial derivatives form the system matrices. This gives us the Linear Time-Invariant (LTI) normal form
#
# ```math
# \begin{aligned}
#   \frac{d\delta\mathbf{x}}{dt} &= \mathbf{A}\,\delta \mathbf{x} + \mathbf{B}\,\delta\mathbf{S}\\
#   \mathbf{\Theta} &= \mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{x}
# \end{aligned}
# ```
#
# This LTI could be the basis for further model reduction emthods, like Balanced Truncation.
#
# To integrate this linearized model with the original PowerDynamics framework,
# we need to provide a wrapper that translates between the original voltage-current
# variables and our new power-phase formulation:
#
# ```math
# \begin{aligned}
#   \delta\mathbf{S} &= \mathbf{i}^*\left(\exp\left(\mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{x}\right)\right) - \mathbf{S}_0\\
#   \frac{d\mathbf{x}}{dt} &= \mathbf{A}\,\delta \mathbf{x} + \mathbf{B}\,\delta\mathbf{S}\\
#   \mathbf{u} &= \exp\left(\mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{x}\right)
# \end{aligned}
# ```
#
# From these equations, we observe that the output voltage $\mathbf{u}$ depends only on the
# internal state deviations $\delta\mathbf{x}$ and the initial phase angles $\mathbf{\Theta}_0$.
# This means that under a global phase shift (rotation of all phase angles by the same amount),
# only the initial phase $\mathbf{\Theta}_{0,i}$ needs to be updated—making it the only covariant variable.
#
# To demonstrate and validate this rotational invariance property, we use the IEEE 9-bus test system.

using PowerDynamics
using NormalFormIdentification
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve
using Graphs
using CairoMakie
using BenchmarkTools

# ## Setting up the IEEE 9-Bus Test System
#
# First, we load the IEEE 9-Bus system from PowerDynamics and compute the steady-state powerflow:

include(joinpath(pkgdir(PowerDynamics), "test", "testsystems.jl"))
nw = TestSystems.load_ieee9bus()

pfnw = powerflow_model(nw)
pf0 = NWState(pfnw)
pfs = find_fixpoint(pfnw, pf0)
nothing # hide

# To test the rotational invariance of our linearization approach, we create a **second powerflow state**
# where we rotate the slack bus reference angle by 30 degrees. This represents a global phase shift
# of the entire system.

pf0_rot = NWState(pfnw)
pf0_rot[VPIndex(1, :slack₊δ)] = deg2rad(30)
pfs_rot = find_fixpoint(pfnw, pf0_rot)
nothing #hide

# Comparing the two powerflow results confirms that they differ only by the applied global phase shift,
# while all other system properties remain identical:

show_powerflow(pfs)

show_powerflow(pfs_rot)

# ## Reference Simulation with Nonlinear Models
#
# To establish a reference solution, we simulate a line trip contingency using the full nonlinear models.
# This will serve as our benchmark for comparing the linearized model accuracy.
#
# We initialize the network from both powerflow states (with and without phase shift) and simulate
# the system response over 10 seconds.

# Perturbation: line failure of 4=>6 at t=1s
deactivate_line = ComponentAffect([], [:pibranch₊active]) do u, p, ctx
    p[:pibranch₊active] = 0
end
cb = PresetTimeComponentCallback([1.0], deactivate_line)
set_callback!(nw[EIndex(4=>6)], cb)

# steady states for both rotated and non-rotated case
s0 = initialize_from_pf(nw; pfs=pfs)
s0_rot = initialize_from_pf(nw; pfs=pfs_rot)

# simulation of both rotated and non-rotated case
prob = ODEProblem(nw, uflat(s0), (0.0, 10.0), copy(pflat(s0)), callback=get_callbacks(nw))
sol = solve(prob, Rodas5P());

prob_rot = ODEProblem(nw, uflat(s0_rot), (0.0, 10.0), copy(pflat(s0_rot)), callback=get_callbacks(nw))
sol_rot = solve(prob_rot, Rodas5P());

# Verify that the solutions differ by exactly the phase shift
@assert sol_rot(0; idxs=VIndex(1,:busbar₊u_arg)) - sol(0; idxs=VIndex(1,:busbar₊u_arg)) ≈ deg2rad(30)

# Plotting results
let
    fig = Figure(size=(600,800));

    # Active power at selected buses
    ax = Axis(fig[1, 1]; title="Active Power", xlabel="Time [s]", ylabel="Power [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), alpha=0.5)
        lines!(ax, sol_rot; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linestyle=:dash)
    end

    # Voltage magnitude at all buses
    ax = Axis(fig[2, 1]; title="Voltage Magnitude", xlabel="Time [s]", ylabel="Voltage [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), alpha=0.5)
        lines!(ax, sol_rot; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linestyle=:dash)
    end

    fig
end

# As expected, both voltage magnitudes and active power profiles are identical for both cases.
# This confirms that these quantities are indeed invariant under global phase shifts.
#
# ## Complete System Linearization
#
# Now we proceed to linearize all dynamic components around their respective operating points.
# This involves computing the Normal Form linearization for each component using the steady-state
# conditions we established above.

# Helper function to extract component state from network state
function get_component_state(s::NWState, cidx)
    nw = extract_nw(s)
    comp = nw[cidx]
    allsym = vcat(sym(comp), psym(comp), insym(comp), outsym(comp))
    Dict(sym => s[VIndex(cidx.compidx, sym)] for sym in allsym)
end
nothing #hide

# We apply the Normal Form linearization procedure to each of the 9 bus components,
# computing the linearized dynamics around their respective operating points:

vms_lin = map(1:9) do i
    nonlinear_model = nw[VIndex(i)]
    comp_state = get_component_state(s0, VIndex(i))
    nf_linearization(nonlinear_model, comp_state)
end;
nw_lin = Network(nw; vertexm=vms_lin)

# The linearized network is then initialized using both the original and phase-shifted powerflow states.
# By simulating both cases, we can verify whether the linearized models preserve the rotational invariance
# property of the original nonlinear system:

s0_lin = initialize_from_pf(nw_lin; pfs=pfs);
prob_lin = ODEProblem(nw_lin, uflat(s0_lin), (0.0, 10.0), copy(pflat(s0_lin)), callback=get_callbacks(nw_lin))
sol_lin = solve(prob_lin, Rodas5P());

s0_lin_rot = initialize_from_pf(nw_lin; pfs=pfs_rot);
prob_lin_rot = ODEProblem(nw_lin, uflat(s0_lin_rot), (0.0, 10.0), copy(pflat(s0_lin_rot)), callback=get_callbacks(nw_lin))
sol_lin_rot = solve(prob_lin_rot, Rodas5P());

let
    fig = Figure(size=(600,800));

    # Active power at selected buses
    ax = Axis(fig[1, 1]; title="Active Power", xlabel="Time [s]", ylabel="Power [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), alpha=0.3)
        lines!(ax, sol_lin; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linestyle=:dash)
        lines!(ax, sol_lin_rot; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linewidth=0.5)
    end

    # Voltage magnitude at all buses
    ax = Axis(fig[2, 1]; title="Voltage Magnitude", xlabel="Time [s]", ylabel="Voltage [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), alpha=0.3)
        lines!(ax, sol_lin; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linestyle=:dash)
        lines!(ax, sol_lin_rot; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linewidth=0.5)
    end

    fig
end

# The plot displays three overlapping traces for each bus (same color per bus):
# - **Transparent solid line**: Original nonlinear model
# - **Dashed line**: Linearized model initialized from non-rotated powerflow
# - **Thin solid line**: Linearized model initialized from rotated powerflow
#
# The agreement between the two normal furm curves demostrate, that the linearization
# in invariant coordinates keeps the global phase shift symmetry.

# ## Excursion: Rotational Symmetry Analysis
#
# In the original generator model, the machine angle δ is covariant (it transforms along with global phase shifts).
# We can verify this numerically by applying a rotation to all relevant variables and
# checking that the steady-state residual remains zero:

rotational_symmetry(nw[VIndex(1)], get_component_state(s0, VIndex(1));
    covariant=[:generator₊machine₊δ])

# In the linearized Normal Form model, the covariant variable is the initial phase angle Θ₀_i, while the rest of the
# states remain invariant.

rotational_symmetry(nw_lin[VIndex(1)], get_component_state(s0_lin, VIndex(1));
    covariant=[:Θ₀_i])

# ## Selective Linearization: Generators Only
#
# In the previous example, we linearized all system components, including simple Kirchhoff junction buses
# and PQ load buses that have relatively simple dynamics. For comparison, let's now create a hybrid model
# where we linearize only the three generator buses (which have complex dynamics) while keeping
# the remaining buses in their original nonlinear form.

vms_lin2 = map(1:9) do i
    if i ∈  [1,2,3]
        nonlinear_model = nw[VIndex(i)]
        comp_state = get_component_state(s0, VIndex(i))
        nf_linearization(nonlinear_model, comp_state)
    else
        copy(nw[VIndex(i)])
    end
end;
nw_lin2 = Network(nw; vertexm=vms_lin2)
s0_lin2 = initialize_from_pf(nw_lin2; pfs=pfs);
prob_lin2 = ODEProblem(nw_lin2, uflat(s0_lin2), (0.0, 10.0), copy(pflat(s0_lin2)), callback=get_callbacks(nw_lin2))
sol_lin2 = solve(prob_lin2, Rodas5P());

let
    fig = Figure(size=(600,800));

    # Active power at selected buses
    ax = Axis(fig[1, 1]; title="Active Power", xlabel="Time [s]", ylabel="Power [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), alpha=0.3)
        lines!(ax, sol_lin; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linestyle=:dash)
        lines!(ax, sol_lin2; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linewidth=0.5)
    end

    # Voltage magnitude at all buses
    ax = Axis(fig[2, 1]; title="Voltage Magnitude", xlabel="Time [s]", ylabel="Voltage [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), alpha=0.3)
        lines!(ax, sol_lin; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linestyle=:dash)
        lines!(ax, sol_lin2; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linewidth=0.5)
    end

    fig
end

# This selective linearization approach yields results that are even closer to the
# full nonlinear solution. This suggests that linearizing only the components with significant
# dynamic complexity can provide an optimal balance between computational efficiency and accuracy.

# ## Computational Performance Analysis
#
# Finally, let's quantify the computational benefits of linearization by benchmarking
# the different model variants. We'll measure both the single function evaluation time
# and the complete ODE solve performance.

dx = zeros(dim(nw)); x = copy(uflat(s0)); p = copy(pflat(s0));
@benchmark $nw($dx, $x, $p, 0.0) seconds=1

dx = zeros(dim(nw_lin)); x = copy(uflat(s0_lin)); p = copy(pflat(s0_lin));
@benchmark $nw_lin($dx, $x, $p, 0.0) seconds=1

dx = zeros(dim(nw_lin2)); x = copy(uflat(s0_lin2)); p = copy(pflat(s0_lin2));
@benchmark $nw_lin2($dx, $x, $p, 0.0) seconds=1

# Next, we benchmark the complete ODE solution process:

@benchmark solve($prob, Rodas5P()) seconds=1

@benchmark solve($prob_lin, Rodas5P()) seconds=1

@benchmark solve($prob_lin2, Rodas5P()) seconds=1

# The results here are... inconclusive. For this system there is no real benefit in linearizing.

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
