# Utilities for simulation of single sigma network
include("simu_utils.jl")

"""
Generate the chemical reactions
for a network of two sigma factor circuits
(ts stands for time sharing, which is not actually guaranteed
"""
function duo_ts_system()
    # model definition
    sigma_model = @reaction_network begin
        # sigma factor 1
        1/τ₁ * β₁ * v₀, ∅ --> σ₁
        1/τ₁ * β₁ * duo_input_hill(Rσ₁, A₁, KS₁, KD₁, n₁), ∅ --> σ₁
        1/τ₁, σ₁ --> ∅
        #1/τ₂ * Rσ₁, ∅ --> A₁
        1/τ₂ * β₁ * v₀, ∅ --> A₁
        1/τ₂ * β₁ * duo_input_hill(Rσ₁, A₁, KS₁, KD₁, n₁), ∅ --> A₁
        1/τ₂, A₁ --> ∅
        # sigma factor 2
        1/τ₁ * β₂ * v₀, ∅ --> σ₂
        1/τ₁ * β₂ * duo_input_hill(Rσ₂, A₂, KS₂, KD₂, n₂), ∅ --> σ₂
        1/τ₁, σ₂ --> ∅
        #1/τ₂ * Rσ₂, ∅ --> A₂
        1/τ₂ * β₂ * v₀, ∅ --> A₂
        1/τ₂ * β₂ * duo_input_hill(Rσ₂, A₂, KS₂, KD₂, n₂), ∅ --> A₂
        1/τ₂, A₂ --> ∅
        # sharing of RNAP
        k₁₁, R + σ₁ --> Rσ₁
        k₂₁, Rσ₁ --> R + σ₁
        k₁₂, R + σ₂ --> Rσ₂
        k₂₂, Rσ₂ --> R + σ₂
        
    end v₀ β₁ β₂ KS₁ KD₁ KS₂ KD₂ n₁ n₂ τ₁ τ₂ k₁₁ k₂₁ k₁₂ k₂₂ η    # η is for SDE only
    
    return sigma_model
end

"""
Similar to simu_all() function, but for duo-time-sharing system
TODO: can merge with simu_all()
"""
function simu_duo_ts_all(_m; _Rₜ = 50, _v₀ = 1e-2, _β₁ = 100., _β₂ = 100.,
        _KS₁ = 0.2, _rK₁ = 1., _KS₂ = 0.2, _rK₂ = 1.,
        _n₁ = 3., _n₂ = 3., _τ₁ = 10., _rτ = 1., 
        _k₁₁ = 1e-3, _k₂₁ = 1e-2, _k₁₂ = 1e-3, _k₂₂ = 1e-2, _η = .1, 
        max_t = 2000., stress_t = 500., plot_max_t = 2000., saveat = 1.0,
        method = "ssa", show_tc = true, show_pp = true, show_hill = true,
        quiet = false)
    # k₁₁ is the *association* rate of σ₁ and k₂₁ is the *dissociation* rate of σ_2
    # similarly we have k₁₂ and k₂₂
    # default time of the induction of stress is increased
    # to set Rσ and σ to steady state
    
    tspan = (0., max_t)    # time course
    # initial state, σ₁, A₁, σ₂, A₂, R, Rσ₁, Rσ₂
    u₀ = [0, 0, 0, 0, _Rₜ, 0, 0]

    # parameters, v₀, β, S, D, n, τ₁, τ₂, k₁, k₂, η
    # S is initially set to 0 and subject to a step change
    p = [_v₀, _β₁, _β₂, _KS₁, _rK₁ * _KS₁, _KS₂, _rK₂ * _KS₂, _n₁, _n₂, _τ₁, _τ₁ * _rτ, 
        _k₁₁, _k₂₁, _k₁₂, _k₂₂, _η]
    
    # choose different methods
    if method == "ssa"
        sol = simu_ssa(_m, tspan, u₀, p, stress_t, _saveat = saveat, dual = true)
    elseif method == "ode"
        sol = simu_ode(_m, tspan, u₀, p, stress_t, _saveat = saveat)
    elseif method == "sde"
        sol = simu_sde(_m, tspan, u₀, p, stress_t, _saveat = saveat)
    else
        println("Simulation method $method not found. Abort.")
        return nothing
    end
    
    # print parameters
    if !quiet
        println("KD₁/KS₁ = ", _rK₁, "; KS₁ = ", _KS₁, 
            "; KD₂/KS₂ = ", _rK₂, "; KS₂ = ", _KS₂, "; τ₂/τ₁ = ", _rτ,
            "; β₁ = ", _β₁, "; β₂ = ", _β₂, "; n₁ = ", _n₁, "; n₂ = ", _n₂, 
            "; k₁₁ = ", _k₁₁, "; k₂₁ = ", _k₂₁, "; k₁₂ = ", _k₁₂, "; k₂₂ = ", _k₂₂,
            "; Rₜ = ", _Rₜ)
    end
    
    # plot
    if !quiet
        if show_tc
            display(plot_timecourse(select_species(sol, [6, 2]), 
                    stress_t, max_t = plot_max_t, labels = ["Rσ₁(t)", "A₁(t)"]))    # σ₁ and A₁
            display(plot_timecourse(select_species(sol, [8, 2]), 
                    stress_t, max_t = plot_max_t, labels = ["σ₁ total", "A₁(t)"]))
            display(plot_timecourse(select_species(sol, [1, 6]), 
                    stress_t, max_t = plot_max_t, labels = ["σ₁ free", "Rσ₁"]))
            display(plot_timecourse(select_species(sol, [6, 7]), 
                    stress_t, max_t = plot_max_t, labels = ["Rσ₁(t)", "Rσ₂(t)"]))    # σ₁ and σ₂
            display(plot_timecourse(select_species(sol, [8, 9]), 
                    stress_t, max_t = plot_max_t, labels = ["σ₁ total", "σ₂ total"]))
            display(plot_timecourse(select_species(sol, [5, 10]), 
                    stress_t, max_t = plot_max_t, labels = ["R free", "R total"]))
        end
        if show_pp
            display(plot_phase_plane(select_species(sol, [6, 2])))    # σ₁ and A₁
            display(plot_phase_plane(select_species(sol, [6, 7])))    # σ₁ and σ₂
        end
        if show_hill
            display(plot_hill(select_species(sol, [1, 2]), 
                    1 / _KS, 1 / (_rK * _KS), _n))
        end
    end
    
    # return arrays
    return sol
end

module My
    struct My_solution
        u::Array{Vector{Float64}, 1}
        t::Array{Float64, 1}
    end
end

"""
Select some of the species
1. σ₁, 2. A₁, 3. σ₂, 4. A₂, 5. R, 6. Rσ₁, 7. Rσ₂
8. σ₁ + Rσ₁, 9. σ₂ + Rσ₂, 10. R + Rσ₁ + Rσ₂
"""
function select_species(sol, vals)
    u_ext = ones(Float64, length(sol.u), 10)
    for i = 1:length(sol.u)
        u_ext[i, :] .= [sol.u[i]; 
            sol.u[i][1] + sol.u[i][6]; 
            sol.u[i][3] + sol.u[i][7];
            sol.u[i][5] + sol.u[i][6] + sol.u[i][7]]
    end
    new_u = [ u_ext[i, vals] for i = 1:size(u_ext, 1) ]
            
#     u_ext = [ [sol.u[i]; sol.u[i][1] + sol.u[i][6]; sol.u[i][3] + sol.u[i][7];
#         sol.u[i][5] + sol.u[i][6] + sol.u[i][7]] for i = 1:length(sol.u) ]
#     new_u = [ u_ext[i][vals] for i = 1:length(u_ext) ]
#     println(typeof(new_u[1]))
    new_sol = My.My_solution(new_u, sol.t)
    return new_sol
end