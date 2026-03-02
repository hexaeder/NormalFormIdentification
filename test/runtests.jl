using Test
using NormalFormIdentification
using NetworkDynamics
using PowerDynamics
using PowerDynamics.Library
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt

@testset "HW VertexModel round-trip" begin
    # Build a simple droop inverter model (from docs/examples/droop_identification.jl)
    @mtkmodel DroopInverter begin
        @components begin
            terminal = Terminal()
        end
        @parameters begin
            Pset, [description="Active power setpoint", guess=1]
            Qset, [description="Reactive power setpoint", guess=0]
            Vset, [description="Voltage setpoint", guess=1]
            ω₀=0, [description="Nominal frequency"]
            Kp=1.0, [description="Active power droop coefficient"]
            Kq=1.0, [description="Reactive power droop coefficient"]
            τ = 1.0, [description="Power filter time constant"]
        end
        @variables begin
            Pmeas(t), [description="Active power measurement", guess=1]
            Qmeas(t), [description="Reactive power measurement", guess=0]
            Pfilt(t), [description="Filtered active power", guess=1]
            Qfilt(t), [description="Filtered reactive power", guess=1]
            ω(t)=1, [description="Frequency"]
            δ(t)=0, [description="Voltage angle", guess=0]
            V(t)=1, [description="Voltage magnitude"]
        end
        @equations begin
            Pmeas ~  terminal.u_r*terminal.i_r + terminal.u_i*terminal.i_i
            Qmeas ~ -terminal.u_r*terminal.i_i + terminal.u_i*terminal.i_r
            τ * Dt(Pfilt) ~ Pmeas - Pfilt
            τ * Dt(Qfilt) ~ Qmeas - Qfilt
            ω ~ ω₀ - Kp * (Pfilt - Pset)
            V ~ Vset - Kq * (Qfilt - Qset)
            Dt(δ) ~ ω - ω₀
            terminal.u_r ~ V*cos(δ)
            terminal.u_i ~ V*sin(δ)
        end
    end

    @named inverter = DroopInverter()
    mtkbus = MTKBus(inverter)
    vm = compile_bus(mtkbus)
    set_default!(vm, :busbar₊u_r, 1.0)
    set_default!(vm, :busbar₊u_i, 0.0)
    set_default!(vm, :busbar₊i_r, -1.0)
    set_default!(vm, :busbar₊i_i, 0.0)
    initialize_component!(vm)

    # Linearize with normal form (HW) transformation
    hwm = hammerstein_wiener_linearization(vm, NormalFormTransformation)

    @testset "HWM metadata" begin
        @test haskey(hwm.metadata, :nl_insym)
        @test haskey(hwm.metadata, :nl_outsym)
        @test haskey(hwm.metadata, :original_vm)
        @test haskey(hwm.metadata, :p0)
    end

    # Convert to VertexModel
    vm_lin = VertexModel(hwm)


    @testset "Steady state" begin
        @test init_residual(vm_lin) < 1e-8
    end

    @testset "Output at operating point" begin
        state_orig = NetworkDynamics.get_defaults_or_inits_dict(vm)
        y_nl0_orig = Float64[state_orig[s] for s in outsym(vm)]

        state_lin = NetworkDynamics.get_defaults_or_inits_dict(vm_lin)
        y_nl0_lin = Float64[state_lin[s] for s in outsym(vm_lin)]

        @test y_nl0_orig ≈ y_nl0_lin atol=1e-10
    end
end
