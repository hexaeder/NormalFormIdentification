using NormalFormIdentification
using NetworkDynamics
using PowerDynamics
using PowerDynamics.Library
using LinearAlgebra
using ForwardDiff
using FiniteDiff
using CairoMakie
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt
using Symbolics

####
#### Analyse a droop inverter model
####
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

@named inverter = DroopInverter()
mtkbus = MTKBus(inverter)
vm = Bus(mtkbus)

# M x' = f(x, i)
# u = f(x)
vm.metadata[:equations]
vm.metadata[:outputeqs]

vm.metadata[:observed]

function _get_symbolic(vm, names)
    sys = vm.metadata[:odesystem]
    allsyms = Symbolics.wrap.(vcat(
        parameters(sys),
        unknowns(sys),
        [eq.lhs for eq in observed(sys)],
    ))
    map(names) do name
        idxs = findall(s -> Symbolics.getname(Symbolics.unwrap(s)) == name, allsyms)
        allsyms[only(idxs)]
    end
end
i_symbolic = _get_symbolic(vm, insym(vm))
u_symbolic = _get_symbolic(vm, outsym(vm))
x_symbolic = _get_symbolic(vm, sym(vm))
p_symbolic = _get_symbolic(vm, psym(vm))

obs_subs = [eq.lhs => eq.rhs for eq in vm.metadata[:observed]]
eqs = Symbolics.simplify(
    Symbolics.fixpoint_sub(vm.metadata[:equations], obs_subs)
)
outeqs = Symbolics.simplify(
    Symbolics.fixpoint_sub(vm.metadata[:outputeqs], obs_subs)
)
break

eqs

####
#### Input transformation
####
@parameters P0, Q0
@variables ΔP(t), ΔQ(t)
ic = conj( P0+ΔP + im*(Q0+ΔQ) / (u_symbolic[1] + im*u_symbolic[2]) )
# ic = conj(ΔP + im*(ΔQ)) / (u_symbolic[1] + im*u_symbolic[2])
i_subs = Dict(i_symbolic .=> [simplify(real(ic)), simplify(imag(ic))])

####
#### Output transformation
####
@variables ϑ(t), ν(t)
complex_phase_eqs = [
   # ν ~ 0.5*log(u_symbolic[1]^2 + u_symbolic[2]^2),
   ν ~ log(p_symbolic[5]),
   # ν ~ atan(u_symbolic[2], u_symbolic[1]),
   ϑ ~ x_symbolic[1], # delta
]
u_subs = Dict(eq.lhs => eq.rhs for eq in outeqs)

####
#### x -> x0+Δx
####
ΔX0vec = map(x_symbolic) do x
  raw_name = ModelingToolkit.getname(x)
  x0_name = Symbol(raw_name, "₀")
  Δx_name = Symbol(replace(string(raw_name), r"₊"=>"₊Δ"))
  x0 = Symbolics.variable(x0_name)
  Δx = Symbolics.variable(Δx_name; T=Symbolics.FnType)(t)
  (x0, Δx)
end
X0s = [ΔX0[1] for ΔX0 in ΔX0vec]
ΔXs = [ΔX0[2] for ΔX0 in ΔX0vec]
x0_subs = Dict(x_symbolic .=> X0s .+ ΔXs)

# f_eqs = Symbolics.simplify(
#     Symbolics.fixpoint_sub(eqs, i_subs)
# )
f_eqs = Symbolics.fixpoint_sub(eqs, merge(i_subs, x0_subs))
f_rhs = [eq.rhs for eq in f_eqs]

# g_eqs = Symbolics.simplify(
#     Symbolics.fixpoint_sub(complex_phase_eqs, u_subs)
# )
g_eqs = Symbolics.fixpoint_sub(complex_phase_eqs, merge(u_subs, x0_subs))
g_rhs = [eq.rhs for eq in g_eqs]

# linearization point
# f_around_ΔQP0 = Symbolics.simplify(Symbolics.fixpoint_sub(f_eqs, ΔQP0_subs))

# f_around_x0 = Symbolics.simplify(Symbolics.fixpoint_sub(f_eqs, x0_subs))

# Symbolics.jacobian([eq.rhs for eq in f_around_ΔQP0], x_symbolic; simplify=true)
# Symbolics.jacobian([eq.rhs for eq in f_around_x0], [ΔQ, ΔP]; simplify=true)
# Symbolics.jacobian([eq.rhs for eq in g_eqs], x_symbolic; simplify=true)


A = Symbolics.jacobian(f_rhs, ΔXs; simplify=false)
B = Symbolics.jacobian(f_rhs, [ΔQ, ΔP]; simplify=false)
C = Symbolics.jacobian(g_rhs, ΔXs; simplify=false)

steadystate_subs = [
    ΔQ => 0,
    ΔP => 0,
    (ΔXs .=> 0)...
]

A = Symbolics.fixpoint_sub(A, steadystate_subs) |> Symbolics.simplify
B = Symbolics.fixpoint_sub(B, steadystate_subs) |> Symbolics.simplify
C = Symbolics.fixpoint_sub(C, steadystate_subs) |> Symbolics.simplify

Symbolics.get_variables.(A)

initialize_subs = [
    X0s[1] => atan(u_symbolic[2], u_symbolic[1]), # delta
    X0s[2] => p_symbolic[4], # Pfilt
    X0s[3] => p_symbolic[2], # Qfilt
]
A = Symbolics.fixpoint_sub(A, initialize_subs) |> Symbolics.simplify
B = Symbolics.fixpoint_sub(B, initialize_subs) |> Symbolics.simplify
C = Symbolics.fixpoint_sub(C, initialize_subs) |> Symbolics.simplify

















# p_subs = []
# ΔQP0_subs = [ΔQ => 0, ΔP => 0]
steadystate_subs = [
    P0 => 0,
    Q0 => 0,
    ΔQ => 0,
    ΔP => 0,
    x_symbolic[1] => atan(u_symbolic[2], u_symbolic[1]), # delta
    x_symbolic[2] => p_symbolic[4], # Pfilt
    x_symbolic[3] => p_symbolic[2], # Pfilt
    u_symbolic[1]^2 + u_symbolic[2]^2 => p_symbolic[5]^2,
]

    ΔQ => 0,
    ΔP => 0,

S_init = conj(Complex(i_symbolic...)) * Complex(u_symbolic...)
steadystate_subs = [
    P0 => p_symbolic[4],
    Q0 => p_symbolic[2],
    ΔP => 0,
    ΔQ => 0,
    # p_symbolic[4] => -P0,
    # p_symbolic[2] => -Q0,
    # x_symbolic[1] => atan(u_symbolic[2], u_symbolic[1]), # delta
    x_symbolic[1] => atan(u_symbolic[2], u_symbolic[1]), # delta
    # x_symbolic[1] => 0, # delta
    x_symbolic[2] => p_symbolic[4], # Pfilt
    x_symbolic[3] => p_symbolic[2], # Qfilt
    # u_symbolic[1]^2 + u_symbolic[2]^2 => p_symbolic[5]^2,
    p_symbolic[5] => sqrt(u_symbolic[1]^2 + u_symbolic[2]^2),
]
Symbolics.simplify(Symbolics.fixpoint_sub(A, steadystate_subs), expand=true)
Symbolics.simplify(Symbolics.fixpoint_sub(B, steadystate_subs), expand=true)
Symbolics.simplify(Symbolics.fixpoint_sub(C, steadystate_subs), expand=true)


























set_default!(vm, :busbar₊u_r, 1.0)
set_default!(vm, :busbar₊u_i, 0.0)
set_default!(vm, :busbar₊i_r, -1.0)
set_default!(vm, :busbar₊i_i, 0.0)
initialize_component!(vm)

dump_initial_state(vm, obs=false)
subs = Dict(map(eq -> eq.lhs => eq.rhs, vm.metadata[:observed]))
feqs = ModelingToolkit.fixpoint_sub(vm.metadata[:equations], subs)
geqs = ModelingToolkit.fixpoint_sub(vm.metadata[:outputeqs], subs)

for eq in geqs
    s = repr(eq)
    s = replace(s, "inverter₊" => "")
    s = replace(s, "Differential(t)" => "Dt")
    s = replace(s, "(t)" => "")
    println(s)
end

lti = get_LTI(vm)

lti.A
lti.B
lti.C




# defining equations
Pmeas ~ terminal.u_r*terminal.i_r + terminal.u_i*terminal.i_i
Qmeas ~ terminal.u_r*terminal.i_i - terminal.u_i*terminal.i_r
τ * Dt(Pfilt) ~ Pmeas - Pfilt
τ * Dt(Qfilt) ~ Qmeas - Qfilt
ω ~ ω₀ - Kp * (Pfilt - Pset)
V ~ Vset - Kq * (Qfilt - Qset)
Dt(δ) ~ ω - ω₀
terminal.u_r ~ V*cos(δ)
terminal.u_i ~ V*sin(δ)

# f equations
Dt(Pfilt) ~ (Pfilt + (Vset + Kq*(Qset - Qfilt))*sin(δ)*busbar₊i_i + (Vset + Kq*(Qset - Qfilt))*busbar₊i_r*cos(δ)) / (-τ)
Dt(Qfilt) ~ (Qfilt + (-Vset + Kq*(-Qset + Qfilt))*sin(δ)*busbar₊i_r + (Vset + Kq*(Qset - Qfilt))*cos(δ)*busbar₊i_i) / (-τ)
    Dt(δ) ~ Kp*(Pset - Pfilt)

# g equations
busbar₊u_r ~ (Vset + Kq*(Qset - Qfilt))*cos(δ)
busbar₊u_i ~ (Vset + Kq*(Qset - Qfilt))*sin(δ)

# linearisierungspunkt (inklusive parameter values)
Inputs:
  busbar₊i_i            =  0
  busbar₊i_r            = -1
States:
  inverter₊Pfilt        =  1          (guess  1)
  inverter₊Qfilt        =  0          (guess  1)
  inverter₊δ            =  0          (guess  0)
Outputs:
  busbar₊u_i            =  0
  busbar₊u_r            =  1
Parameters:
  inverter₊Kp           =  1
  inverter₊Kq           =  1
  inverter₊Pset         =  1          (guess  1)
  inverter₊Qset         =  0          (guess  0)
  inverter₊Vset         =  1          (guess  1)
  inverter₊τ            =  1
  inverter₊ω₀ (unused)  =  0

# linearisiertes system
# input:  [ΔQ, ΔP]
# states: [Pfilt, Qfilt, δ]
# output: [ϑ_r, ϑ_i] oder [Δϑ_r, Δϑ_i]?
julia> lti.A
 -1.0  -0.0  -0.0
 -0.0  -1.0  -0.0
 -1.0   0.0   0.0

julia> lti.B
  0.0   1.0
 -1.0  -0.0
  0.0   0.0

julia> lti.C
 0.0  -1.0  0.0
 0.0   0.0  1.0







unsolvable_identity(x) = x
@register_symbolic unsolvable_identity(x)
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
        C = 1, [description="Kapazität der internen Impedanz", guess=1]
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
        i_r(t), [description="Strom intern", guess=1]
        i_i(t), [description="Strom intern", guess=0]
    end
    @equations begin
        i_r ~ (u_r - terminal.u_r) / R
        i_i ~ (u_i - terminal.u_i) / R
        Dt(u_r) ~  ω0*u_i + 1/C * (terminal.i_r + i_r)
        Dt(u_i) ~ -ω0*u_r + 1/C * (terminal.i_i + i_i)


        Pmeas ~  u_r*terminal.i_r + u_i*terminal.i_i
        Qmeas ~ -u_r*terminal.i_i + u_i*terminal.i_r
        τ * Dt(Pfilt) ~ Pmeas - Pfilt
        τ * Dt(Qfilt) ~ Qmeas - Qfilt
        ω ~ ω₀ - Kp * (Pfilt - Pset) ## Frequenz senken, wenn P höher als Sollwert
        # NOTE: the sign of Kq here is not fully clear!!
        V ~ Vset - Kq * (Qfilt - Qset) ## Spannung senken, wenn Q höher als Sollwert
        Dt(δ) ~ ω - ω₀
        u_r ~ V*cos(δ)
        u_i ~ V*sin(δ)
    end
end;
@named inverter = DroopInverterResistance()
mtkbus = MTKBus(inverter)
vm = Bus(mtkbus)
set_default!(vm, :busbar₊u_r, 1.0)
set_default!(vm, :busbar₊u_i, 0.0)
set_default!(vm, :busbar₊i_r, -1.0)
set_default!(vm, :busbar₊i_i, 0.0)
initialize_component!(vm)

dump_initial_state(vm, obs=false)
subs = Dict(map(eq -> eq.lhs => eq.rhs, vm.metadata[:observed]))
feqs = ModelingToolkit.fixpoint_sub(vm.metadata[:equations], subs)
geqs = ModelingToolkit.fixpoint_sub(vm.metadata[:outputeqs], subs)

for eq in geqs
    s = repr(eq)
    s = replace(s, "inverter₊" => "")
    s = replace(s, "Differential(t)" => "Dt")
    s = replace(s, "(t)" => "")
    println(s)
end

lti = get_LTI(vm)
lti.A
lti.B
lti.C

# defining equations
terminal.i_r ~ (u_r - terminal.u_r) / R
terminal.i_i ~ (u_i - terminal.u_i) / R
Pmeas ~  u_r*terminal.i_r + u_i*terminal.i_i
Qmeas ~ -u_r*terminal.i_i + u_i*terminal.i_r
τ * Dt(Pfilt) ~ Pmeas - Pfilt
τ * Dt(Qfilt) ~ Qmeas - Qfilt
ω ~ ω₀ - Kp * (Pfilt - Pset)
V ~ Vset - Kq * (Qfilt - Qset)
Dt(δ) ~ ω - ω₀
u_r ~ V*cos(δ)
u_i ~ V*sin(δ)

# f equations
Dt(Pfilt) ~ (Pfilt + (Vset + Kq*(Qset - Qfilt))*sin(δ)*busbar₊i_i + (Vset + Kq*(Qset - Qfilt))*busbar₊i_r*cos(δ)) / (-τ)
Dt(Qfilt) ~ (Qfilt + (Vset + Kq*(Qset - Qfilt))*sin(δ)*busbar₊i_r - (Vset + Kq*(Qset - Qfilt))*cos(δ)*busbar₊i_i) / (-τ)
Dt(δ) ~ Kp*(Pset - Pfilt)
0 ~ busbar₊u_r - R*(busbar₊i_r + ((Vset + Kq*(Qset - Qfilt))*cos(δ)) / R)
0 ~ busbar₊u_i - R*(((Vset + Kq*(Qset - Qfilt))*sin(δ)) / R + busbar₊i_i)

# g equations
busbar₊u_r ~ busbar₊u_r
busbar₊u_i ~ busbar₊u_i

# linearisierungspunkt (inklusive parameter values)
Inputs:
  busbar₊i_i            =  0
  busbar₊i_r            = -1
States:
  busbar₊u_i            =  0
  busbar₊u_r            =  1
  inverter₊Pfilt        =  2          (guess  1)
  inverter₊Qfilt        =  0          (guess  1)
  inverter₊δ            =  0          (guess  0)
Outputs:
  busbar₊u_i            =  0
  busbar₊u_r            =  1
Parameters:
  inverter₊Kp           =  1
  inverter₊Kq           =  1
  inverter₊Pset         =  2          (guess  1)
  inverter₊Qset         =  0.5        (guess  0)
  inverter₊R            =  1
  inverter₊Vset         =  1.5        (guess  1)
  inverter₊τ            =  1
  inverter₊ω₀ (unused)  =  0

# lineares system
# input:  [ΔQ, ΔP]
# states: [Pfilt, Qfilt, δ, busbar₊u_r, busbar₊u_i]
# output: [ϑ_r, ϑ_i] oder [Δϑ_r, Δϑ_i]?
julia> lti.A
 -1.0  -1.0   0.0  -2.0   0.0
  0.0  -1.0   2.0   0.0  -2.0
 -1.0   0.0   0.0   0.0   0.0
  0.0   1.0   0.0   0.0   0.0
  0.0   0.0  -2.0   0.0   2.0

julia> lti.B
  0.0  -2.0
 -2.0   0.0
  0.0   0.0
  0.0  -1.0
  1.0   0.0

julia> lti.C
 0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0







@named classical = Library.ClassicalMachine(τ_m_input=false)
mtkbus = MTKBus(classical)
vm = Bus(mtkbus)
set_default!(vm, :busbar₊u_r, 1.0)
set_default!(vm, :busbar₊u_i, 0.0)
set_default!(vm, :busbar₊i_r, -1.0)
set_default!(vm, :busbar₊i_i, 0.0)

initialize_component!(vm, verbose=true)
dump_initial_state(vm)
vm.sym

sys_droop = get_LTI(vm)
sys_droop.A
sys_droop.B
sys_droop.C
# sys_droop.G(0.000*im)
# sys_droop.G(0.00010*im)
# sys_droop.G(0.01*im)
# sys_droop.G(1*im) ./ im

# sys_droop.A
# sys_droop.B
# sys_droop.C

A = sys_droop.A
A[2,2] = 0.0
B = sys_droop.B
C = sys_droop.C
M = sys_droop.M
G = s -> s * C * inv(s*M - A) * B

G(0*im)
G(0.00000001*im)
G(0.001*im)
G(0.01*im)
G(0.1*im)

sym(vm)

bode_plot(G; inout=(1,1))
bode_plot(G; inout=(2,2))
