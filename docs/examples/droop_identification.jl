#=
# Droop Identification Example
=#
using NormalFormIdentification
using NetworkDynamics
using PowerDynamics
using PowerDynamics.Library
using CairoMakie
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt

#=
## Basic droop inverter model
=#
@mtkmodel DroopInverter begin
    @components begin
        terminal = Terminal()
    end
    @parameters begin
        Pset, [description="Wirkleistungs-Sollwert", guess=1]
        Qset, [description="Blindleistungs-Sollwert", guess=0]
        Vset, [description="Spannungs-Sollwert", guess=1]
        ω₀=0, [description="Nennfrequenz"]
        Kp=1.0, [description="Wirkleistungs-Droop-Koeffizient"]
        Kq=1.0, [description="Blindleistungs-Droop-Koeffizient"]
        τ = 1.0, [description="Zeitkonstante des Leistungsfilters"]
    end
    @variables begin
        Pmeas(t), [description="Wirkleistungsmessung", guess=1]
        Qmeas(t), [description="Blindleistungsmessung", guess=0]
        Pfilt(t), [description="Gefilterte Wirkleistung", guess=1]
        Qfilt(t), [description="Gefilterte Blindleistung", guess=1]
        ω(t)=1, [description="Frequenz"]
        δ(t)=0, [description="Spannungswinkel", guess=0]
        V(t)=1, [description="Spannungsbetrag"]
    end
    @equations begin
        Pmeas ~  terminal.u_r*terminal.i_r + terminal.u_i*terminal.i_i
        Qmeas ~ -terminal.u_r*terminal.i_i + terminal.u_i*terminal.i_r
        τ * Dt(Pfilt) ~ Pmeas - Pfilt
        τ * Dt(Qfilt) ~ Qmeas - Qfilt
        ω ~ ω₀ - Kp * (Pfilt - Pset) ## Frequenz senken, wenn P höher als Sollwert
        V ~ Vset - Kq * (Qfilt - Qset) ## Spannung senken, wenn Q höher als Sollwert
        Dt(δ) ~ ω - ω₀
        terminal.u_r ~ V*cos(δ)
        terminal.u_i ~ V*sin(δ)
    end
end;

#=
Build the bus model
=#
@named inverter = DroopInverter()
mtkbus = MTKBus(inverter)
vm = Bus(mtkbus)
nothing #hide
# in order to get the LTI we need to initialize the system
set_default!(vm, :busbar₊u_r, 1.0)
set_default!(vm, :busbar₊u_i, 0.0)
set_default!(vm, :busbar₊i_r, -1.0)
set_default!(vm, :busbar₊i_i, 0.0)
initialize_component!(vm)


#=
Now we have a fully initialzied model and can inspectit further
=#
print_equations(vm; remove_ns=[:inverter])
#-
print_linearization(vm)
#=
Next we can generate the bode plots for
```math
G(s) = C \, \left(s\,M -A\right)^{-1}\,B
```
=#
bode_plot(get_LTI(vm).G)
#=
Also the slighly adapted Gs
```math
G_s(s) = s\cdot G(s) = s \cdot C \, \left(s\,M -A\right)^{-1}\,B
```
=#
bode_plot(get_LTI(vm).Gs)


#=
## Droop Inverter behind resistance
=#
@mtkmodel DroopInverterResistance begin
    @components begin
        terminal = Terminal()
    end
    @parameters begin
        Pset, [description="Wirkleistungs-Sollwert", guess=1]
        Qset, [description="Blindleistungs-Sollwert", guess=0]
        Vset, [description="Spannungs-Sollwert", guess=1]
        ω₀=0, [description="Nennfrequenz"]
        Kp=1.0, [description="Wirkleistungs-Droop-Koeffizient"]
        Kq=1.0, [description="Blindleistungs-Droop-Koeffizient"]
        τ = 1.0, [description="Zeitkonstante des Leistungsfilters"]
        R = 1, [description="Widerstand der internen Impedanz", guess=1]
    end
    @variables begin
        Pmeas(t), [description="Wirkleistungsmessung", guess=1]
        Qmeas(t), [description="Blindleistungsmessung", guess=0]
        Pfilt(t), [description="Gefilterte Wirkleistung", guess=1]
        Qfilt(t), [description="Gefilterte Blindleistung", guess=1]
        ω(t)=1, [description="Frequenz"]
        δ(t)=0, [description="Spannungswinkel", guess=0]
        V(t)=1, [description="Spannungsbetrag"]
        u_r(t), [description="Spannung intern", guess=1]
        u_i(t), [description="Spannung intern", guess=0]
    end
    @equations begin
        terminal.i_r ~ (u_r - terminal.u_r) / R
        terminal.i_i ~ (u_i - terminal.u_i) / R
        Pmeas ~  u_r*terminal.i_r + u_i*terminal.i_i
        Qmeas ~ -u_r*terminal.i_i + u_i*terminal.i_r
        τ * Dt(Pfilt) ~ Pmeas - Pfilt
        τ * Dt(Qfilt) ~ Qmeas - Qfilt
        ω ~ ω₀ - Kp * (Pfilt - Pset) ## Frequenz senken, wenn P höher als Sollwert
        V ~ Vset - Kq * (Qfilt - Qset) ## Spannung senken, wenn Q höher als Sollwert
        Dt(δ) ~ ω - ω₀
        u_r ~ V*cos(δ)
        u_i ~ V*sin(δ)
    end
end;

#=
Now let's build and analyze this model with internal resistance.
=#
@named inverter = DroopInverterResistance()
mtkbus = MTKBus(inverter)
vm = Bus(mtkbus)
set_default!(vm, :busbar₊u_r, 1.0)
set_default!(vm, :busbar₊u_i, 0.0)
set_default!(vm, :busbar₊i_r, -1.0)
set_default!(vm, :busbar₊i_i, 0.0)
initialize_component!(vm)

#=
Let's examine the equations and linearization of this model:
=#
print_equations(vm; remove_ns=[:inverter])
#-
print_linearization(vm)
#=
Next we can generate the bode plots for
```math
G(s) = C \, \left(s\,M -A\right)^{-1}\,B
```
=#
bode_plot(get_LTI(vm).G)
#=
Also the slighly adapted Gs
```math
G_s(s) = s\cdot G(s) = s \cdot C \, \left(s\,M -A\right)^{-1}\,B
```
=#
bode_plot(get_LTI(vm).Gs)
