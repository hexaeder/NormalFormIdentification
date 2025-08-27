#=
# Linearization of Models in IEEE 9-Bus System

In this example, we'll substitute dynamic models with their
Normal Form (NF) linearization and validate that the linearized models remain accurate
under global phase shifts.

In PowerDynamics.jl, we start with a nonlinear dynamical system of the form

```math
\begin{aligned}
  \frac{d\mathbf{x}}{dt} &= f\left(\mathbf{x}, \mathbf{i}\right)\\
  \mathbf{u} &= g(\mathbf{x})
\end{aligned}
```
where $\mathbf{u}$ is the complex voltage output and $\mathbf{i}$ is the complex current input.

However, for linearization it's more convenient to work with
power-phase variables rather than voltage-current variables. To transform this into the
normal form convention $\mathbf{\Delta S} \mapsto \mathbf{\Delta \Theta}$,
we need to reformulate the system as
```math
\begin{aligned}
  \frac{d\mathbf{x}}{dt} &= f\left(\mathbf{x}, \left(\frac{\mathbf{S}}{g(\mathbf{x})}\right)^*\right) &&= \bar{f}(\mathbf{x}, \mathbf{S})\\
  \mathbf{\Theta} &= \mathrm{ln}\,g(\mathbf{x}) &&= \bar{g}(\mathbf{x})
\end{aligned}
```

Next, we identify the equilibrium point around which we want to linearize the system:

```math
\begin{aligned}
  \frac{d\mathbf{x}}{dt} &= 0 = f(\mathbf{x}_0, \mathbf{i}_0)\\
    \mathbf{u}_0 &= g(\mathbf{x}_0)\\
  \mathbf{S}_0 &= \mathbf{u}_0 \cdot \mathbf{i}_0^*\\
  \mathbf{\Theta}_0 &= \mathrm{ln}\,g(\mathbf{x}_0) = \mathrm{ln}\,\mathbf{u}_0
\end{aligned}
```
The linearization around the equilibrium point looks liek this:
```math
\begin{aligned}
  \frac{d\mathbf{x}}{dt} &= \underbrace{\bar{f}(\mathbf{x}_0, \mathbf{S}_0)}_{=0} + \frac{\partial \bar{f}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_0, \mathbf{S}_0}\delta \mathbf{x} +
  \frac{\partial \bar{f}}{\partial \mathbf{S}}\bigg|_{\mathbf{x}_0, \mathbf{S}_0}\delta\mathbf{S}\\
  \mathbf{\Theta} &= \mathbf{\Theta}_0 + \frac{\partial \bar{g}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_0} \delta \mathbf{x}
\end{aligned}
```
where the partial derivatives form the system matrices. This gives us the Linear Time-Invariant (LTI) normal form

```math
\begin{aligned}
  \frac{d\delta\mathbf{x}}{dt} &= \mathbf{A}\,\delta \mathbf{x} + \mathbf{B}\,\delta\mathbf{S}\\
  \mathbf{\Theta} &= \mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{x}
\end{aligned}
```

This LTI could be the basis for further model reduction emthods, like Balanced Truncation.

To integrate this linearized model with the original PowerDynamics framework,
we need to provide a wrapper that translates between the original voltage-current
variables and our new power-phase formulation:

```math
\begin{aligned}
  \delta\mathbf{S} &= \mathbf{i}^*\left(\exp\left(\mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{x}\right)\right) - \mathbf{S}_0\\
  \frac{d\mathbf{x}}{dt} &= \mathbf{A}\,\delta \mathbf{x} + \mathbf{B}\,\delta\mathbf{S}\\
  \mathbf{u} &= \exp\left(\mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{x}\right)
\end{aligned}
```

From these equations, we observe that the output voltage $\mathbf{u}$ depends only on the
internal state deviations $\delta\mathbf{x}$ and the initial phase angles $\mathbf{\Theta}_0$.
This means that under a global phase shift (rotation of all phase angles by the same amount),
only the initial phase $\mathbf{\Theta}_{0,i}$ needs to be updated—making it the only covariant variable.

To demonstrate and validate this rotational invariance property, we use the IEEE 9-bus test system.
=#
using PowerDynamics
using NormalFormIdentification
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve
using Graphs
using CairoMakie
using BenchmarkTools

#=
## Setting up the IEEE 9-Bus Test System

First, we load the IEEE 9-Bus system from PowerDynamics and compute the steady-state powerflow:
=#
include(joinpath(pkgdir(PowerDynamics), "test", "testsystems.jl"))
nw = TestSystems.load_ieee9bus()

pfnw = powerflow_model(nw)
pf0 = NWState(pfnw)
pfs = find_fixpoint(pfnw, pf0)
nothing # hide

#=
To test the rotational invariance of our linearization approach, we create a **second powerflow state**
where we rotate the slack bus reference angle by 30 degrees. This represents a global phase shift
of the entire system.
=#

pf0_rot = NWState(pfnw)
pf0_rot[VPIndex(1, :slack₊δ)] = deg2rad(30)
pfs_rot = find_fixpoint(pfnw, pf0_rot)
nothing #hide
#=
Comparing the two powerflow results confirms that they differ only by the applied global phase shift,
while all other system properties remain identical:
=#
show_powerflow(pfs)
#-
show_powerflow(pfs_rot)
#=
## Reference Simulation with Nonlinear Models

To establish a reference solution, we simulate a line trip contingency using the full nonlinear models.
This will serve as our benchmark for comparing the linearized model accuracy.

We initialize the network from both powerflow states (with and without phase shift) and simulate
the system response over 10 seconds.
=#

## Perturbation: line failure of 4=>6 at t=1s
deactivate_line = ComponentAffect([], [:pibranch₊active]) do u, p, ctx
    p[:pibranch₊active] = 0
end
cb = PresetTimeComponentCallback([1.0], deactivate_line)
set_callback!(nw[EIndex(4=>6)], cb)

## steady states for both rotated and non-rotated case
s0 = initialize_from_pf(nw; pfs=pfs)
s0_rot = initialize_from_pf(nw; pfs=pfs_rot)

## simulation of both rotated and non-rotated case
prob = ODEProblem(nw, uflat(s0), (0.0, 10.0), copy(pflat(s0)), callback=get_callbacks(nw))
sol = solve(prob, Rodas5P());

prob_rot = ODEProblem(nw, uflat(s0_rot), (0.0, 10.0), copy(pflat(s0_rot)), callback=get_callbacks(nw))
sol_rot = solve(prob_rot, Rodas5P());

## Verify that the solutions differ by exactly the phase shift
@assert sol_rot(0; idxs=VIndex(1,:busbar₊u_arg)) - sol(0; idxs=VIndex(1,:busbar₊u_arg)) ≈ deg2rad(30)

#=
```@raw html
<script> (function() {const thisScript = document.currentScript; setTimeout(function() {let current = thisScript.nextElementSibling; while (current) {const code = current.querySelector('code'); if (code) {const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Show code'; const parent = code.parentNode; parent.parentNode.insertBefore(details, parent); details.appendChild(summary); details.appendChild(parent); break;} current = current.nextElementSibling;}}, 100);})(); </script>
```
=#
let
    fig = Figure(size=(600,800));

    ## Active power at selected buses
    ax = Axis(fig[1, 1]; title="Active Power", xlabel="Time [s]", ylabel="Power [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), alpha=0.5)
        lines!(ax, sol_rot; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linestyle=:dash)
    end

    ## Voltage magnitude at all buses
    ax = Axis(fig[2, 1]; title="Voltage Magnitude", xlabel="Time [s]", ylabel="Voltage [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), alpha=0.5)
        lines!(ax, sol_rot; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linestyle=:dash)
    end

    fig
end

#=
As expected, both voltage magnitudes and active power profiles are identical for both cases.
This confirms that these quantities are indeed invariant under global phase shifts.

## Complete System Linearization

Now we proceed to linearize all dynamic components around their respective operating points.
This involves computing the Normal Form linearization for each component using the steady-state
conditions we established above.
=#
## Helper function to extract component state from network state
function get_component_state(s::NWState, cidx)
    nw = extract_nw(s)
    comp = nw[cidx]
    allsym = vcat(sym(comp), psym(comp), insym(comp), outsym(comp))
    Dict(sym => s[VIndex(cidx.compidx, sym)] for sym in allsym)
end
nothing #hide

#=
We apply the Normal Form linearization procedure to each of the 9 bus components,
computing the linearized dynamics around their respective operating points:
=#
vms_lin = map(1:9) do i
    nonlinear_model = nw[VIndex(i)]
    comp_state = get_component_state(s0, VIndex(i))
    nf_linearization(nonlinear_model, comp_state)
end;
nw_lin = Network(nw; vertexm=vms_lin)

#=
The linearized network is then initialized using both the original and phase-shifted powerflow states.
By simulating both cases, we can verify whether the linearized models preserve the rotational invariance
property of the original nonlinear system:
=#
s0_lin = initialize_from_pf(nw_lin; pfs=pfs);
prob_lin = ODEProblem(nw_lin, uflat(s0_lin), (0.0, 10.0), copy(pflat(s0_lin)), callback=get_callbacks(nw_lin))
sol_lin = solve(prob_lin, Rodas5P());

s0_lin_rot = initialize_from_pf(nw_lin; pfs=pfs_rot);
prob_lin_rot = ODEProblem(nw_lin, uflat(s0_lin_rot), (0.0, 10.0), copy(pflat(s0_lin_rot)), callback=get_callbacks(nw_lin))
sol_lin_rot = solve(prob_lin_rot, Rodas5P());
nothing #hide
#=
```@raw html
<script> (function() {const thisScript = document.currentScript; setTimeout(function() {let current = thisScript.nextElementSibling; while (current) {const code = current.querySelector('code'); if (code) {const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Show code'; const parent = code.parentNode; parent.parentNode.insertBefore(details, parent); details.appendChild(summary); details.appendChild(parent); break;} current = current.nextElementSibling;}}, 100);})(); </script>
```
=#

let
    fig = Figure(size=(600,800));

    ## Active power at selected buses
    ax = Axis(fig[1, 1]; title="Active Power", xlabel="Time [s]", ylabel="Power [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), alpha=0.3)
        lines!(ax, sol_lin; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linestyle=:dash)
        lines!(ax, sol_lin_rot; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linewidth=0.5)
    end

    ## Voltage magnitude at all buses
    ax = Axis(fig[2, 1]; title="Voltage Magnitude", xlabel="Time [s]", ylabel="Voltage [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), alpha=0.3)
        lines!(ax, sol_lin; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linestyle=:dash)
        lines!(ax, sol_lin_rot; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linewidth=0.5)
    end

    fig
end


#=
The plot displays three overlapping traces for each bus (same color per bus):
- **Transparent solid line**: Original nonlinear model
- **Dashed line**: Linearized model initialized from non-rotated powerflow
- **Thin solid line**: Linearized model initialized from rotated powerflow

The agreement between the two normal furm curves demostrate, that the linearization
in invariant coordinates keeps the global phase shift symmetry.
=#

#=
## Excursion: Rotational Symmetry Analysis

In the original generator model, the machine angle δ is covariant (it transforms along with global phase shifts).
We can verify this numerically by applying a rotation to all relevant variables and
checking that the steady-state residual remains zero:
=#
rotational_symmetry(nw[VIndex(1)], get_component_state(s0, VIndex(1));
    covariant=[:generator₊machine₊δ])
#=
In the linearized Normal Form model, the covariant variable is the initial phase angle Θ₀_i, while the rest of the
states remain invariant.
=#
rotational_symmetry(nw_lin[VIndex(1)], get_component_state(s0_lin, VIndex(1));
    covariant=[:Θ₀_i])

#=
## Selective Linearization: Generators Only

In the previous example, we linearized all system components, including simple Kirchhoff junction buses
and PQ load buses that have relatively simple dynamics. For comparison, let's now create a hybrid model
where we linearize only the three generator buses (which have complex dynamics) while keeping
the remaining buses in their original nonlinear form.
=#
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
nothing #hide
#=
```@raw html
<script> (function() {const thisScript = document.currentScript; setTimeout(function() {let current = thisScript.nextElementSibling; while (current) {const code = current.querySelector('code'); if (code) {const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Show code'; const parent = code.parentNode; parent.parentNode.insertBefore(details, parent); details.appendChild(summary); details.appendChild(parent); break;} current = current.nextElementSibling;}}, 100);})(); </script>
```
=#
let
    fig = Figure(size=(600,800));

    ## Active power at selected buses
    ax = Axis(fig[1, 1]; title="Active Power", xlabel="Time [s]", ylabel="Power [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), alpha=0.3)
        lines!(ax, sol_lin; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linestyle=:dash)
        lines!(ax, sol_lin2; idxs=VIndex(i,:busbar₊P), label="Bus $i", color=Cycled(i), linewidth=0.5)
    end

    ## Voltage magnitude at all buses
    ax = Axis(fig[2, 1]; title="Voltage Magnitude", xlabel="Time [s]", ylabel="Voltage [pu]")
    for i in 1:9
        lines!(ax, sol; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), alpha=0.3)
        lines!(ax, sol_lin; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linestyle=:dash)
        lines!(ax, sol_lin2; idxs=VIndex(i,:busbar₊u_mag), label="Bus $i", color=Cycled(i), linewidth=0.5)
    end

    fig
end
#=
This selective linearization approach yields results that are even closer to the
full nonlinear solution. This suggests that linearizing only the components with significant
dynamic complexity can provide an optimal balance between computational efficiency and accuracy.
=#

#=
## Bonus: Inspection of Inernal States

By the power of observables, we can still inspect "estimated" internal states wheras
```math
\mathbf{x}_{\mathrm{original}} \approx \mathbf{x}_0 + \delta \mathbf{x}
```
The results are significantly improved if the voltage explicitly appears in the state vector.
In this case, the "correct" voltage is inserted in the state vector as a basis for
the estimation of the other states/observables.

```@raw html
<script> (function() {const thisScript = document.currentScript; setTimeout(function() {let current = thisScript.nextElementSibling; while (current) {const code = current.querySelector('code'); if (code) {const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Show code'; const parent = code.parentNode; parent.parentNode.insertBefore(details, parent); details.appendChild(summary); details.appendChild(parent); break;} current = current.nextElementSibling;}}, 100);})(); </script>
```
=#
let
    fig = Figure(size=(600,800));
    i=1

    ax = Axis(fig[1, 1]; title="Rotor Angle", xlabel="Time [s]", ylabel="angel [rad]")
    lines!(ax, sol; idxs=VIndex(i,:generator₊machine₊δ), color=Cycled(1), alpha=0.3)
    lines!(ax, sol_lin; idxs=VIndex(i,:estim₊generator₊machine₊δ), color=Cycled(1), linestyle=:dash)

    ax = Axis(fig[2, 1]; title="Transient Voltage d-axis", xlabel="Time [s]", ylabel="Voltage [pu]")
    lines!(ax, sol; idxs=VIndex(i,:generator₊machine₊E′_d), color=Cycled(1), alpha=0.3)
    lines!(ax, sol_lin; idxs=VIndex(i,:estim₊generator₊machine₊E′_d), color=Cycled(1), linestyle=:dash)

    ax = Axis(fig[3, 1]; title="Transient Voltage q-axis", xlabel="Time [s]", ylabel="Voltage [pu]")
    lines!(ax, sol; idxs=VIndex(i,:generator₊machine₊E′_q), color=Cycled(1), alpha=0.3)
    lines!(ax, sol_lin; idxs=VIndex(i,:estim₊generator₊machine₊E′_q), color=Cycled(1), linestyle=:dash)

    fig
end


#=
## Computational Performance Analysis
```@raw html
<details>
<summary>Click to expand!</summary>
```

Finally, let's quantify the computational benefits of linearization by benchmarking
the different model variants. We'll measure both the single function evaluation time
and the complete ODE solve performance.
=#

dx = zeros(dim(nw)); x = copy(uflat(s0)); p = copy(pflat(s0));
@benchmark $nw($dx, $x, $p, 0.0) seconds=1
#-

dx = zeros(dim(nw_lin)); x = copy(uflat(s0_lin)); p = copy(pflat(s0_lin));
@benchmark $nw_lin($dx, $x, $p, 0.0) seconds=1
#-
dx = zeros(dim(nw_lin2)); x = copy(uflat(s0_lin2)); p = copy(pflat(s0_lin2));
@benchmark $nw_lin2($dx, $x, $p, 0.0) seconds=1

#=
Next, we benchmark the complete ODE solution process:
=#
@benchmark solve($prob, Rodas5P()) seconds=1
#-
@benchmark solve($prob_lin, Rodas5P()) seconds=1
#-
@benchmark solve($prob_lin2, Rodas5P()) seconds=1

#=
The results here are... inconclusive. For this system there is no real benefit in linearizing.

```@raw html
</details>
```
=#

#=
## Resolve constaints
```math
\begin{aligned}
  \mathbf{M}\frac{d\delta\mathbf{x}}{dt} &= \mathbf{A}\,\delta \mathbf{x} + \mathbf{B}\,\delta\mathbf{S}\\
  \mathbf{\Theta} &= \mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{x}
\end{aligned}
```
If we have such a system with a non-singular mass matrix $\mathbf{M}$, we can also resolve the algebraic constraints.
This leads to a non-zero $\mathbf{\bar D}$ matrix in the output equation (i.e. a Feed Forward)
```math
\begin{aligned}
  \frac{d\delta\mathbf{z}}{dt} &= \mathbf{\bar A}\,\delta \mathbf{z} + \mathbf{\bar B}\,\delta\mathbf{S}\\
  \mathbf{\Theta} &= \mathbf{\Theta}_0 + \mathbf{\bar C}\,\delta \mathbf{z} + \mathbf{\bar D}\,\delta\mathbf{S}
\end{aligned}
```

This feed foward is problematic, because we cannot explicitly calculate $\delta\mathbf{S}$ anymore!
```math
\begin{aligned}
  \delta\mathbf{S} &= \mathbf{i}^*\left(\exp\left(\mathbf{\Theta}_0 + \mathbf{\bar C}\,\delta \mathbf{x} + \mathbf{\bar D}\,\delta\mathbf{S}\right)\right) - \mathbf{S}_0\\
  \frac{d\mathbf{x}}{dt} &= \mathbf{\bar A}\,\delta \mathbf{x} + \mathbf{\bar B}\,\delta\mathbf{S}\\
  \mathbf{u} &= \exp\left(\mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{x} + \mathbf{\bar D}\,\delta\mathbf{S}\right)
\end{aligned}
```

We can cirumvent this problem by introducing 2 real contraints for the imaginary and real part of the voltage again:
```math
\begin{aligned}
  \delta\mathbf{S} &= \mathbf{i}^*\mathbf{u} - \mathbf{S}_0\\
  \frac{d\mathbf{z}}{dt} &= \mathbf{\bar A}\,\delta \mathbf{z} + \mathbf{\bar B}\,\delta\mathbf{S}\\
  0 &=\mathbf{u} - \exp\left(\mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{z} + \mathbf{\bar D}\,\delta\mathbf{S}\right)
\end{aligned}
```
With this definition, our VertexModel looks like this:
```math
\begin{aligned}
\begin{bmatrix}
1 &\\
&\ddots\\
&&1\\
&&&0
\end{bmatrix}\frac{d\mathbf{s}}{dt}&=
\begin{bmatrix}
\mathbf{\bar A}\,\delta \mathbf{z} + \mathbf{\bar B}\,\delta\mathbf{S}\\
\mathbf{u} - \exp\left(\mathbf{\Theta}_0 + \mathbf{C}\,\delta \mathbf{z} + \mathbf{\bar D}\,\delta\mathbf{S}\right)
\end{bmatrix}\quad\text{where}\quad\delta\mathbf{S} = \mathbf{i}^*\mathbf{u} - \mathbf{S}_0\quad\text{and}\quad\mathbf{s}=
\begin{bmatrix}\mathbf{\bar z}\\\mathbf{u}\end{bmatrix}
\\
\mathbf{u} &=\begin{bmatrix}
0 &\\
&\ddots\\
&&0\\
&&&1
\end{bmatrix}\mathbf{s}
\end{aligned}
```

We'll apply this linearization to all three generatros and can make sure, that it does not change the systems behavior
by comparing the results for a single generator
=#
GENERATOR = 3
vms_lin_noc = map(1:9) do i
    if i ∈ 1:3
        nonlinear_model = nw[VIndex(i)]
        comp_state = get_component_state(s0, VIndex(i))
        nf_linearization(nonlinear_model, comp_state; transform_constraints=true)
    else
        copy(nw[VIndex(i)])
    end
end;
nw_lin_noc = Network(nw; vertexm=vms_lin_noc)
s0_lin_noc = initialize_from_pf(nw_lin_noc; pfs=pfs);
prob_lin_noc = ODEProblem(nw_lin_noc, uflat(s0_lin_noc), (0.0, 10.0), copy(pflat(s0_lin_noc)), callback=get_callbacks(nw_lin_noc))
sol_lin_noc = solve(prob_lin_noc, Rodas5P());
nothing #hide

#=
```@raw html
<script> (function() {const thisScript = document.currentScript; setTimeout(function() {let current = thisScript.nextElementSibling; while (current) {const code = current.querySelector('code'); if (code) {const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Show code'; const parent = code.parentNode; parent.parentNode.insertBefore(details, parent); details.appendChild(summary); details.appendChild(parent); break;} current = current.nextElementSibling;}}, 100);})(); </script>
```
=#
let
    fig = Figure(size=(600,800));
    i = GENERATOR
    ax = Axis(fig[1, 1]; title="Active Power on Bus $i", xlabel="Time [s]", ylabel="Power [pu]")
    lines!(ax, sol; idxs=VIndex(i,:busbar₊P), color=Cycled(1), label="Reference Solution")
    lines!(ax, sol_lin2; idxs=VIndex(i,:busbar₊P), color=Cycled(2), label="Linearization")
    lines!(ax, sol_lin_noc; idxs=VIndex(i,:busbar₊P), color=Cycled(3), label="Lin with solved Constraints", linestyle=:dash)
    axislegend(ax; position=:rt)

    ax = Axis(fig[2, 1]; title="Voltage Magnitude on Bus $i", xlabel="Time [s]", ylabel="Voltage [pu]")
    lines!(ax, sol; idxs=VIndex(i,:busbar₊u_mag), color=Cycled(1))
    lines!(ax, sol_lin2; idxs=VIndex(i,:busbar₊u_mag), color=Cycled(2))
    lines!(ax, sol_lin_noc; idxs=VIndex(i,:busbar₊u_mag), color=Cycled(3), linestyle=:dash)

    ax = Axis(fig[3, 1]; title="Transient Voltage q-axis on Bus $i", xlabel="Time [s]", ylabel="Voltage [pu]")
    lines!(ax, sol; idxs=VIndex(i,:generator₊machine₊E′_q), color=Cycled(1), alpha=0.3)
    lines!(ax, sol_lin2; idxs=VIndex(i,:estim₊generator₊machine₊E′_q), color=Cycled(2))
    lines!(ax, sol_lin_noc; idxs=VIndex(i,:estim₊generator₊machine₊E′_q), color=Cycled(3), linestyle=:dash)
    fig
end

#=
Note that also the `:estim` states survive the transformations.

For this example, reducing the constraints is not to interesting, because we had 3 befor and now we have 2.
=#
@assert dim(nw[VIndex(GENERATOR)]) == 14 #hide
@assert dim(nw_lin_noc[VIndex(GENERATOR)]) == 13 #hide
println("Original Generator Dimension   = ", dim(nw[VIndex(GENERATOR)]))
println("Linearized Generator Dimension = ", dim(nw_lin_noc[VIndex(GENERATOR)]))
nothing #hide
#=
However it does make a difference when it comes do balanced truncation.
=#

#=
## Balanced Truncation
Lets say we've solve the constaints so our system does not have a sigular mass matrix anymore:
```math
\begin{aligned}
  \frac{d\delta\mathbf{x}}{dt} &= \mathbf{A}\,\delta \mathbf{x} + \mathbf{B}\,\delta\mathbf{S}\\
  \delta\mathbf{\Theta} &= \mathbf{C}\,\delta \mathbf{x} + \mathbf{D}\,\delta \mathbf{S}
\end{aligned}
```
We immediatly encounter the problem, that $\mathbf{A}$ has a zero eigenvalue!
Such a system is not minimal and we cannot apply balanced truncation directly.
Therefore, we'll do a similarity transformation to seperate the integrator:
(in theory my state and matrice would nee som other symbol but we'll ignore that here)
```math
\newcommand{\dx}{\delta \mathbf{x}}
\newcommand{\vb}[1]{\mathbf{#1}}
\newcommand{\dxmat}{\begin{bmatrix}\dx_s\\\dx_m\end{bmatrix}}
\begin{aligned}
  \frac{d}{dt}\dxmat &=\begin{bmatrix}\vb{A}_{ss}&\mathbf{A_{sm}}\\0&0\end{bmatrix}\,\dxmat + \begin{bmatrix}\mathbf{B}_{s}\\\mathbf{B_{m}}\end{bmatrix}\,\delta\mathbf{S}\\
  \delta\mathbf{\Theta} &= \begin{bmatrix}\mathbf{C}_{s}&\mathbf{C_{m}}\end{bmatrix}\,\dxmat +\mathbf{D}\,\delta\mathbf{S}
\end{aligned}
```
So we've split the system into a **stable part (s)** and an **marginal part (m)**.
The expandend form is:
```math
\newcommand{\dx}{\delta\mathbf{x}}
\newcommand{\dS}{\delta\mathbf{S}}
\newcommand{\vb}[1]{\mathbf{#1}}
\newcommand{\dxmat}{\begin{bmatrix}\dx_s\\\dx_m\end{bmatrix}}
\begin{aligned}
  \frac{d}{dt}\dx_s &=\vb A_{ss}\,\dx_s + \vb A_{sm}\,\dx_m + \vb B_s\,\dS\\
  \frac{d}{dt}\dx_m &= \vb B_m\,\dS\\
  \delta\mathbf{\Theta} &= \vb C_s\,\dx_s+\vb C_m\,\dx_m + \vb D\,\dS
\end{aligned}
```
We can reformulate this a bit to find our stable subsystem:
```math
\newcommand{\dx}{\delta\mathbf{x}}
\newcommand{\dS}{\delta\mathbf{S}}
\newcommand{\vb}[1]{\mathbf{#1}}
\newcommand{\dxmat}{\begin{bmatrix}\dx_s\\\dx_m\end{bmatrix}}
\newcommand{\bmatrix}[1]{\begin{bmatrix}#1\end{bmatrix}}
\begin{aligned}
  \frac{d}{dt}\dx_m &= \vb B_m\,\dS\\
  \frac{d}{dt}\dx_s &= \vb A_{ss}\,\dx_s+\bmatrix{\vb B_s&\vb A_{sm}}\bmatrix{\dS\\\dx_m}\\
  \delta\mathbf{\Theta} &= \vb C_s\,\dx_s+\bmatrix{\vb D&\vb C_m}\bmatrix{\dS\\\dx_m}
\end{aligned}
```
Those are the matrices we can throw into the balanced truncation algorithm. We get out a system like this:
```math
\newcommand{\dx}{\delta\mathbf{x}}
\newcommand{\dS}{\delta\mathbf{S}}
\newcommand{\vb}[1]{\mathbf{#1}}
\newcommand{\dxmat}{\begin{bmatrix}\dx_s\\\dx_m\end{bmatrix}}
\newcommand{\bmatrix}[1]{\begin{bmatrix}#1\end{bmatrix}}
\newcommand{\Ared}{\vb{A}^\mathrm{r}}
\newcommand{\Bred}{\vb{B}^\mathrm{r}}
\newcommand{\Cred}{\vb{C}^\mathrm{r}}
\newcommand{\Dred}{\vb{D}^\mathrm{r}}
\newcommand{\xred}{\delta\vb{x}^\mathrm{r}}
\begin{aligned}
  \frac{d}{dt}\dx_m &= \vb B_m\,\dS\\
  \frac{d}{dt}\xred &= \Ared\,\xred+\Bred\bmatrix{\dS\\\dx_m}\\
  \delta\mathbf{\Theta} &= \Cred\,\xred+\Dred\bmatrix{\dS\\\dx_m}
\end{aligned}
```
For reconstructing this, we need to expand it a bit:
```math
\newcommand{\dx}{\delta\mathbf{x}}
\newcommand{\dS}{\delta\mathbf{S}}
\newcommand{\vb}[1]{\mathbf{#1}}
\newcommand{\dxmat}{\begin{bmatrix}\dx_s\\\dx_m\end{bmatrix}}
\newcommand{\bmatrix}[1]{\begin{bmatrix}#1\end{bmatrix}}
\newcommand{\Ared}{\vb{A}^\mathrm{r}}
\newcommand{\Bred}{\vb{B}^\mathrm{r}}
\newcommand{\Cred}{\vb{C}^\mathrm{r}}
\newcommand{\Dred}{\vb{D}^\mathrm{r}}
\newcommand{\xred}{\delta\vb{x}^\mathrm{r}}
\begin{aligned}
  \frac{d}{dt}\dx_m &= \vb B_m\,\dS\\
  \frac{d}{dt}\xred &= \Ared\,\xred+\bmatrix{\Bred_{\dS}&\Bred_{\dx_m}}\bmatrix{\dS\\\dx_m}
&&= \Ared\,\xred+\Bred_{\dS}\,\dS+\Bred_{\dx_m}\,\dx_m\\
  \delta\mathbf{\Theta} &= \Cred\,\xred+\bmatrix{\Dred_{\dS}&\Dred_{\dx_m}}\bmatrix{\dS\\\dx_m}
 &&= \Cred\,\xred+\Dred_{\dS}\,\dS+\Dred_{\dx_m}\,\dx_m
\end{aligned}
```
With that, we can reconstruct a LTI for the full state again:
```math
\newcommand{\dx}{\delta\mathbf{x}}
\newcommand{\dS}{\delta\mathbf{S}}
\newcommand{\vb}[1]{\mathbf{#1}}
\newcommand{\dxmat}{\begin{bmatrix}\dx_s\\\dx_m\end{bmatrix}}
\newcommand{\bmatrix}[1]{\begin{bmatrix}#1\end{bmatrix}}
\newcommand{\Ared}{\vb{A}^\mathrm{r}}
\newcommand{\Bred}{\vb{B}^\mathrm{r}}
\newcommand{\Cred}{\vb{C}^\mathrm{r}}
\newcommand{\Dred}{\vb{D}^\mathrm{r}}
\newcommand{\xred}{\delta\vb{x}^\mathrm{r}}
\begin{aligned}
  \frac{d}{dt}\bmatrix{\xred\\\dx_m} &= \bmatrix{\Ared & \Bred_{\dx_m}\\0&0}\bmatrix{\xred\\\dx_m} + \bmatrix{\Bred_{\dS}\\\vb B_m}\dS\\
\delta\mathbf{\Theta}&=\bmatrix{\Cred&\Dred_{\dx_m}}\bmatrix{\xred\\\dx_m}+\Dred_{\dS}\,\dS
\end{aligned}
```

So lets simulate this for different reduction orders and with/without residualization.
Residualization means, that we assume the trucated states to be in steady state (i.e. singular perturbation approximation).
Without residualization, we assume the trucanted states to be zero.
In general, residualization keeps the dc gain while without residualization, the high frequency regime is more accurate.
=#

REDUCTIONS = [3,4,5,6,7]
sol_bt = let
    _sol_bt = Dict()
    for reduction in REDUCTIONS
        for residualization in (true, false)
            _vms = map(1:9) do i
                if i ∈ 1:3
                    nonlinear_model = nw[VIndex(i)]
                    comp_state = get_component_state(s0, VIndex(i))
                    nf_linearization(nonlinear_model, comp_state;
                        transform_constraints=true,
                        reduction, residualization)
                else
                    copy(nw[VIndex(i)])
                end
            end;
            _nw = Network(nw; vertexm=_vms)
            _s0 = initialize_from_pf(_nw; pfs=pfs, verbose=false);
            _prob = ODEProblem(_nw, uflat(_s0), (0.0, 10.0), copy(pflat(_s0)), callback=get_callbacks(_nw))
            _sol = solve(_prob, Rodas5P());
            ## println("Solved for -$reduction states ", residualization ? "with" : "without", " residualization")
            _sol_bt[(reduction, residualization)] = _sol
        end
    end
    _sol_bt
end;

#=
### Plot of active Power at Bus
```@raw html
<script> (function() {const thisScript = document.currentScript; setTimeout(function() {let current = thisScript.nextElementSibling; while (current) {const code = current.querySelector('code'); if (code) {const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Show code'; const parent = code.parentNode; parent.parentNode.insertBefore(details, parent); details.appendChild(summary); details.appendChild(parent); break;} current = current.nextElementSibling;}}, 100);})(); </script>
```
=#
let
    fig = Figure(size=(1000,1000));
    col = 1
    Label(fig[1, 1], "BT with residualization", tellwidth=false)
    Label(fig[1, 2], "BT without residualization", tellwidth=false)
    for residualization in (true, false)
        for BUS in 1:3
            ax = Axis(fig[BUS+1, col]; xlabel="Time [s]", ylabel="Power Bus $BUS [pu]", xlabelvisible=BUS==3, ylabelvisible=residualization)
            for (i, reduction) in enumerate(reverse(REDUCTIONS))
                thisdim = dim(extract_nw(sol_bt[(reduction, residualization)])[VIndex(BUS)])
                fulldim = dim(nw[VIndex(BUS)])
                lines!(sol_bt[(reduction, residualization)]; idxs=VIndex(BUS,:busbar₊P),
                    color=Cycled(i), label="-$reduction states ($thisdim/$fulldim)",
                    alpha=range(0.5, 1, length=length(REDUCTIONS))[i],
                    linewidth=range(5, 2, length=length(REDUCTIONS))[i])
            end
            lines!(sol; idxs=VIndex(BUS,:busbar₊P), color=:black, label="Reference Solution", linewidth=1.5)
            BUS==1 && residualization && axislegend(ax; position=:rt)
            xlims!(0.9, 10.0)
        end
        col += 1
    end
    fig
end
#=
### Plot of Bus Voltage Magnitude
```@raw html
<script> (function() {const thisScript = document.currentScript; setTimeout(function() {let current = thisScript.nextElementSibling; while (current) {const code = current.querySelector('code'); if (code) {const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Show code'; const parent = code.parentNode; parent.parentNode.insertBefore(details, parent); details.appendChild(summary); details.appendChild(parent); break;} current = current.nextElementSibling;}}, 100);})(); </script>
```
=#
let
    fig = Figure(size=(1000,1000));
    col = 1
    Label(fig[1, 1], "BT with residualization", tellwidth=false)
    Label(fig[1, 2], "BT without residualization", tellwidth=false)
    for residualization in (true, false)
        for BUS in 1:3
            ax = Axis(fig[BUS+1, col]; xlabel="Time [s]", ylabel="Voltage Bus $BUS [pu]", xlabelvisible=BUS==3, ylabelvisible=residualization)
            for (i, reduction) in enumerate(reverse(REDUCTIONS))
                thisdim = dim(extract_nw(sol_bt[(reduction, residualization)])[VIndex(BUS)])
                fulldim = dim(nw[VIndex(BUS)])
                lines!(sol_bt[(reduction, residualization)]; idxs=VIndex(BUS,:busbar₊u_mag),
                    color=Cycled(i), label="-$reduction states ($thisdim/$fulldim)",
                    alpha=range(0.5, 1, length=length(REDUCTIONS))[i],
                    linewidth=range(5, 2, length=length(REDUCTIONS))[i])
            end
            lines!(sol; idxs=VIndex(BUS,:busbar₊u_mag), color=:black, label="Reference Solution", linewidth=1.5)
            BUS==1 && residualization && axislegend(ax; position=:rt)
            xlims!(0.9, 10.0)
        end
        col += 1
    end
    fig
end
