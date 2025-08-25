#=
# Linearization of Models in IEEE 9-Bus System

In this example, we'll subsititute dynamic models with their
NF linearization and check on validity under a global phase shift.

In powerdynamics, we start wit ha sytem of the form
```math
\begin{aligned}
  \dv{t}\vb{x} &= f\left(\vb{x}, \vb{i}\right)\\
  \vb{u} &= g(\vb{x})
\end{aligned}
```
where $\vb{u}$ is the complex voltage output and $\vb{i}$ is the complex current input.

If we want to bring this into the normalform convention $\vb{\Delta S} \mapsto \vb{\Delta \Theta}$ for linearization we need to find
```math
\begin{aligned}
  \dv{t}\vb{x} &= f\left(\vb{x}, \left(\frac{\vb S}{g(\vb{x})}\right)^*\right) &&= \bar{f}(\vb{x}, \vb{S})\\
  \vb{\Theta} &= \mathrm{ln}\,g(\vb{x}) &&= \bar{g}(\vb{x})
\end{aligned}
```

```math
\begin{aligned}
  \dv{t}\vb{x} &= 0 = f(\vb{x}_0, \vb{i}_0)\\
    \vb{u}_0 &= g(\vb{x}_0)\\
  \vb{S}_0 &= \vb{u}_0 \cdot \vb{i}_0^*\\
  \vb{\Theta}_0 &= \mathrm{ln}\,g(\vb{x}_0) = \mathrm{ln}\,\vb{u}_0
\end{aligned}
```
around which we want to linearize.
```math
\begin{aligned}
  \dv{t}\vb{x} &= \cancel{\bar{f}(\xn, \Sn)} + \pdv{\vb{x}}\bar{f}\bigg|_{\xn, \Sn}\delta \vb x +
  \pdv{\vb{S}}\bar{f}\bigg|_{\xn, \Sn}\delta\vb{S}\\
  \vb{\Theta} &= \Thetan + \pdv{\vb x}\bar{g}\bigg|_{\xn} \delta \vb x
\end{aligned}
```
which is the LTI normal form

\begin{aligned}
  \dv{t}\vb{x} &= \vb{A}\,\delta \vb x + \vb{B}\,\delta\vb{S}\\
  \vb{\Theta} &= \Thetan + \vb{C}\,\delta \vb x
\end{aligned}

In order to simulate this system, we need to add a wrapper for the old input-output variables

\begin{aligned}
  \delta\vb{S} &= \vb{i}^*\left(\exp\left(\Thetan + \vb{C}\,\delta \vb x\right)\right) - \vb{S}_0\\
  \dv{t}\vb{x} &= \vb{A}\,\delta \vb x + \vb{B}\,\delta\vb{S}\\
  \vb{\Theta} &= \Thetan + \vb{C}\,\delta \vb x\\
  \vb{u} &= \exp\left(\Thetan + \vb{C}\,\delta \vb x\right)
\end{aligned}

As a test system we use the ieee9 bus system.
=#
using PowerDynamics
using NormalFormIdentification
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve
using Graphs
using BenchmarkTools

#=
## Defining the base network

Lets load the 9-Bus system from PowerDynamics and get a powerflow state:
=#
include(joinpath(pkgdir(PowerDynamics), "test", "testsystems.jl"))
nw = TestSystems.load_ieee9bus()

pfnw = powerflow_model(nw)
pf0 = NWState(pfnw)
pfs = find_fixpoint(pfnw, pf0)
nothing # hide

#=
We creat a **second powerflow state** for which we rotate the slack reference angle by
30 degrees.
=#

pf0_rot = NWState(pfnw)
pf0_rot[VPIndex(1, :slack₊δ)] = deg2rad(30)
pfs_rot = find_fixpoint(pfnw, pf0_rot)
nothing #hide
#=
If we look at the two powerflow results, we see
that they only differ by a global phase shift:
=#
show_powerflow(pfs)
#-
show_powerflow(pfs_rot)
#=
## Reference Simulation (no linearization)
As a reference solution, we simulate a line trip event.

We initialize the network both with and without global phase shift and solve for 10 seconds.
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

## lets make sure that the solutions differ by exactly the phase shift
@assert sol_rot(0; idxs=VIndex(1,:busbar₊u_arg)) - sol(0; idxs=VIndex(1,:busbar₊u_arg)) ≈ deg2rad(30)

## Plotting results
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

#=
We see that both voltage magnitude and active power are identical for both cases, which was expected sice those are
invariant to a global phase shift.

## Linearization of all dynamic components
Next, we linearize all dynamic components around the operating point.
=#
## help function to extract the state of a component from a NWState
function get_component_state(s::NWState, cidx)
    nw = extract_nw(s)
    comp = nw[cidx]
    allsym = vcat(sym(comp), psym(comp), insym(comp), outsym(comp))
    Dict(sym => s[VIndex(cidx.compidx, sym)] for sym in allsym)
end
nothing #hide

#=
For each of the 9 vertex models we get the normal form linearization around the operating point.
=#
vms_lin = map(1:9) do i
    nonlinear_model = nw[VIndex(i)]
    comp_state = get_component_state(s0, VIndex(i))
    nf_linearization(nonlinear_model, comp_state)
end;
nw_lin = Network(nw; vertexm=vms_lin)
#=
The linearized network is then initialized with around the shifted and non-shifted powerflow.
We then solve and inspect the results.
=#
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

#=
Here we see 3 graphs of the same color:
- solid washed out line: nonlinear model
- dashed line: linearized model around non-rotated operating point
- thin solid line: linearized model around rotated operating point

We see, that the linearization is indeed valid for a global phase shift!
=#

#=
## Partial Linearization
In the above example we linearized everything, also the pure kirchhoff junction busses and the pq busses.
As comparison, we now only linearize the 3 generator busses and keep the rest nonlinear.
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
#=
We see, that the partially linearized model is actually closer to the full nonlinear solution.
=#


#=
## Performance Comparison
Lastly, we want to compare the performance of the nonlinear and linearized model.
=#

dx = zeros(dim(nw)); x = copy(uflat(s0)); p = copy(pflat(s0));
@benchmark $nw($dx, $x, $p, 0.0)
#-

dx = zeros(dim(nw_lin)); x = copy(uflat(s0_lin)); p = copy(pflat(s0_lin));
@benchmark $nw_lin($dx, $x, $p, 0.0)
#-
dx = zeros(dim(nw_lin2)); x = copy(uflat(s0_lin2)); p = copy(pflat(s0_lin2));
@benchmark $nw_lin2($dx, $x, $p, 0.0)

#=
Also lets compae the full solve performacne
=#
@benchmark solve($prob, Rodas5P())
#-
@benchmark solve($prob_lin, Rodas5P())
#-
@benchmark solve($prob_lin2, Rodas5P())
