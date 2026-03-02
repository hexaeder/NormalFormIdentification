using PowerDynamics
using PowerDynamics.Library
using DataFrames
using ModelingToolkit
using CairoMakie
using CSV
using Graphs

EXAMPLEDIR = joinpath(pkgdir(PowerDynamics), "docs", "examples")
include(joinpath(EXAMPLEDIR, "ieee39_part1.jl")) # brings nw intos cope
formula = @initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
set_initformula!(nw[VIndex(31)], formula)
set_initformula!(nw[VIndex(39)], formula)
pfs = solve_powerflow(nw)
s0 = initialize_from_pf!(nw; verbose=false, pfs)

vms_nflin = map(1:nv(nw)) do i
    println(i)
    nonlinear_model = nw[VIndex(i)]
    hw_linearization(nonlinear_model, NormalFormTransformation)
end;
# vms_nflin_legacy = map(1:nv(nw)) do i
#     println(i)
#     nonlinear_model = nw[VIndex(i)]
#     nf_linearization(nonlinear_model)
# end;
vms_dqlin = map(1:nv(nw)) do i
    println(i)
    nonlinear_model = nw[VIndex(i)]
    hw_linearization(nonlinear_model, NormalFormIdentification.NoTransformation())
end;

nw_nf_lin = Network(nw; vertexm=vms_nflin)
s0_nf_lin = initialize_from_pf(nw_nf_lin; verbose=false, pfs)

# nw_nf_lin_legacy = Network(nw; vertexm=vms_nflin_legacy)
# s0_nf_lin_legacy = initialize_from_pf(nw_nf_lin_legacy; verbose=false, pfs)

nw_dq_lin = Network(nw; vertexm=vms_dqlin)
s0_dq_lin = initialize_from_pf(nw_dq_lin; verbose=false, pfs)

begin
    # Select the line to be affected by the short circuit
    AFFECTED_LINE = 11
    # Define callback to enable short circuit at t=0.1s
    _enable_short = ComponentAffect([], [:piline₊shortcircuit]) do u, p, ctx
        @info "Short circuit activated on line $(ctx.src)→$(ctx.dst) at t = $(ctx.t)s"
        p[:piline₊shortcircuit] = 1
    end
    shortcircuit_cb = PresetTimeComponentCallback(0.1, _enable_short)

    # Define callback to disconnect line at t=0.2s (fault clearing)
    _disable_line = ComponentAffect([], [:piline₊active]) do u, p, ctx
        @info "Line $(ctx.src)→$(ctx.dst) disconnected at t = $(ctx.t)s"
        p[:piline₊active] = 0
    end
    _enable_line = ComponentAffect([], [:piline₊active]) do u, p, ctx
        # @info "Line $(ctx.src)→$(ctx.dst) connected at t = $(ctx.t)s"
        # p[:piline₊active] = 1
    end
    deactivate_cb = PresetTimeComponentCallback(0.2, _disable_line)
    activate_cb = PresetTimeComponentCallback(1.3, _enable_line)

    add_comp_cb = Dict(EIndex(AFFECTED_LINE) => (shortcircuit_cb, deactivate_cb, activate_cb))

    @info "Run sol"
    prob = ODEProblem(nw, copy(s0), (0.0, 15.0); add_comp_cb)
    sol = solve(prob, Rodas5P())

    @info "Run sol_nf_lin"
    prob_nf_lin = ODEProblem(nw_nf_lin, s0_nf_lin, (0.0, 15.0); add_comp_cb)
    sol_nf_lin = solve(prob_nf_lin, Rodas5P())

    # @info "Run sol_nf_lin_legacy"
    # prob_nf_lin_legacy = ODEProblem(nw_nf_lin_legacy, s0_nf_lin_legacy, (0.0, 15.0); add_comp_cb)
    # sol_nf_lin_legacy = solve(prob_nf_lin_legacy, Rodas5P())

    @info "Run sol_dq_lin"
    prob_dq_lin = ODEProblem(nw_dq_lin, s0_dq_lin, (0.0, 15.0); add_comp_cb)
    sol_dq_lin = solve(prob_dq_lin, Rodas5P())
end;

sol_nf_lin

#


let fig = Figure(; size=(800, 800))
    ts = range(0, 15, length=1000)
    src_bus, dst_bus = get_graphelement(nw[EIndex(AFFECTED_LINE)])

    ax1 = Axis(fig[1, 1];
        title="Voltage Magnitude at Bus $src_bus (source)",
        xlabel="Time [s]",
        ylabel="Voltage Magnitude [pu]")
    lines!(ax1, ts, sol(ts; idxs=VIndex(src_bus, :busbar₊u_mag)).u;
           label="Nonlinear", linewidth=2)
    lines!(ax1, ts, sol_nf_lin(ts; idxs=VIndex(src_bus, :busbar₊u_mag)).u;
           label="NF Linearized", linewidth=2)
    # lines!(ax1, ts, sol_nf_lin_legacy(ts; idxs=VIndex(src_bus, :busbar₊u_mag)).u;
    #        label="NF Linearized (legacy)", linewidth=2)
    lines!(ax1, ts, sol_dq_lin(ts; idxs=VIndex(src_bus, :busbar₊u_mag)).u;
           label="DQ Linearized", linewidth=2)
    axislegend(ax1; position=:rb)

    ax2 = Axis(fig[2, 1];
        title="Voltage Angle at Bus $src_bus (source)",
        xlabel="Time [s]",
        ylabel="Voltage Angle [rad]")
    lines!(ax2, ts, sol(ts; idxs=VIndex(src_bus, :busbar₊u_arg)).u;
           label="Nonlinear", linewidth=2)
    lines!(ax2, ts, sol_nf_lin(ts; idxs=VIndex(src_bus, :busbar₊u_arg)).u;
           label="NF Linearized", linewidth=2)
    # lines!(ax2, ts, sol_nf_lin_legacy(ts; idxs=VIndex(src_bus, :busbar₊u_arg)).u;
    #        label="NF Linearized (legacy)", linewidth=2)
    lines!(ax2, ts, sol_dq_lin(ts; idxs=VIndex(src_bus, :busbar₊u_arg)).u;
           label="DQ Linearized", linewidth=2)
    axislegend(ax2; position=:rb)

    # bus u_r
    ax3 = Axis(fig[3, 1];
        title="Voltage u_i at Bus $src_bus (source)",
        xlabel="Time [s]",
        ylabel="Voltage Angle [rad]")
    lines!(ax3, ts, sol(ts; idxs=VIndex(src_bus, :busbar₊u_i)).u;
           label="Nonlinear", linewidth=2)
    lines!(ax3, ts, sol_nf_lin(ts; idxs=VIndex(src_bus, :busbar₊u_i)).u;
           label="NF Linearized", linewidth=2)
    # lines!(ax3, ts, sol_nf_lin_legacy(ts; idxs=VIndex(src_bus, :busbar₊u_i)).u;
    #        label="NF Linearized (legacy)", linewidth=2)
    lines!(ax3, ts, sol_dq_lin(ts; idxs=VIndex(src_bus, :busbar₊u_i)).u;
           label="DQ Linearized", linewidth=2)
    axislegend(ax2; position=:rb)


    fig
end
