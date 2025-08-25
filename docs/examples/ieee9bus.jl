using PowerDynamics
using NormalFormIdentification
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve
using Graphs
using BenchmarkTools


include(joinpath(pkgdir(PowerDynamics), "test", "testsystems.jl"))
nw = TestSystems.load_ieee9bus()

pfnw = powerflow_model(nw)
pf0 = NWState(pfnw)
pfs = find_fixpoint(pfnw, pf0)

pf0_rot = NWState(pfnw)
pf0_rot[VPIndex(1, :slack₊δ)] = deg2rad(30)
pfs_rot = find_fixpoint(pfnw, pf0_rot)

show_powerflow(pfs)
show_powerflow(pfs_rot)

s0 = initialize_from_pf(nw; pfs=pfs)
s0_rot = initialize_from_pf(nw; pfs=pfs_rot)

deactivate_line = ComponentAffect([], [:pibranch₊active]) do u, p, ctx
    p[:pibranch₊active] = 0
end
cb = PresetTimeComponentCallback([1.0], deactivate_line)
set_callback!(nw[EIndex(4=>6)], cb)
nw[EIndex(4=>6)] # hide

prob = ODEProblem(nw, uflat(s0), (0.0, 10.0), copy(pflat(s0)), callback=get_callbacks(nw))
sol = solve(prob, Rodas5P());

prob_rot = ODEProblem(nw, uflat(s0_rot), (0.0, 10.0), copy(pflat(s0_rot)), callback=get_callbacks(nw))
sol_rot = solve(prob_rot, Rodas5P());

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
Perfect, matches expectation!

## Linearized Version
=#
function get_component_state(s::NWState, cidx)
    nw = extract_nw(s)
    comp = nw[cidx]
    allsym = vcat(sym(comp), psym(comp), insym(comp), outsym(comp))
    Dict(sym => s[VIndex(cidx.compidx, sym)] for sym in allsym)
end

vms_lin = map(1:9) do i
    nonlinear_model = nw[VIndex(i)]
    comp_state = get_component_state(s0, VIndex(i))
    nf_linearization(nonlinear_model, comp_state)
end;

nw_lin = Network(nw; vertexm=vms_lin)
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
Less complete linearization
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
