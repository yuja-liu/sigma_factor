using Catalyst, Plots, StochasticDiffEq, DiffEqJump, Latexify
using DifferentialEquations
using Random
using Statistics

#Random.seed!(7777)

###############
# model definition
###############
"duo-input Hill function. X and Y are the
activational and repressional inputs respectively"
duo_input_hill(X, Y, S, D, n) = 
    (S * X)^n / ((S * X)^n + (D * Y)^n + 1)

"Generate the general sigma factor model reactions"
function reaction_system(; _δ = 1)
    # model definition
    sigma_model = @reaction_network begin
        1/τ₁ * β * v₀, ∅ --> σ
        1/τ₁ * β * duo_input_hill(σ, A, S, D, n), ∅ --> σ
        1/τ₁, σ --> ∅
        1/τ₂ * σ, ∅ --> A
        1/τ₂, A --> ∅
    end v₀ β S D n τ₁ τ₂ η    # η is for SDE only
    
    # if δ is specified, change the step size
    if(_δ > 1)
        # modify the stoich of σ prod
        @parameters t β S D n τ₁
        @variables σ(t) A(t)
        ## the A prod eqn
        sigma_model.eqs[2] = 
            Reaction(1/τ₁ * β * duo_input_hill(σ, A, S, D, n), nothing, [σ], nothing, [_δ])
        println("burst size δ = ", _δ)
    end
    
    return sigma_model
end

function interpret_sol(sol)
    t_vec = sol.t
    σ_vec = [sol.u[i][1] for i in 1:length(t_vec)]
    A_vec = [sol.u[i][2] for i in 1:length(t_vec)]
    
    # provisional convert to int
    if typeof(t_vec) == Array{Float64, 1}
        t_vec = Int64.(round.(t_vec))
        σ_vec = Int64.(round.(σ_vec))
        A_vec = Int64.(round.(A_vec))
    end
    
    return t_vec, σ_vec, A_vec
end

"plot time course of the simulation"
function plot_timecourse(sol, stress_t; max_t = 2000., show_vars = [1, 2])
    t, σ, A = interpret_sol(sol)
    traj = [σ A]
    if maximum(sol.t) > max_t
        println("Warning: max timestep for plotting exceeded. Not showing entire time course")
        traj = traj[t .<= max_t, :]
        t = t[t .<= max_t]
    end
    plt = plot(t, traj[:, show_vars],
        ylabel = "# molecules",
        title = "Time course",
        labels = ["σ(t)" "A(t)"][show_vars],
        alpha = 0.7)
    
    plot!(plt, [stress_t], seriestype = "vline", color = "red", 
        linestyle = :dash, labels = "Adding stress")
    return plt
end

###############
# plotting
###############
# TODO: do we want to normalize σ and A?    probably no
"plot phase plane"
function plot_phase_plane(sol; max_t = 2000.0)
    t_vec, σ_vec, A_vec = interpret_sol(sol)
    # truncate
    if maximum(t_vec) > max_t
        println("Warning: max time point for phase plane plotting excceed. Truncated")
        σ_vec = σ_vec[t_vec .<= max_t]
        A_vec = A_vec[t_vec .<= max_t]
    end
    
    # phase plane plot
    plt = plot(σ_vec, A_vec, labels = "phase plane", legend = :topleft)
    plot!(plt, σ_vec, A_vec, seriestype = :scatter,
        markerstrokewidth = 0, alpha = 0.2, color = "black")
    plot!(plt, [0, maximum(A_vec)], [0, maximum(A_vec)], 
        color = "red", linestyle =:dash, 
        labels = "y = x", title = "Phase Plane",
        xlabel = "σ", ylabel = "A")
#     plot!(pp_plot, _S / _D * [0, maximum(A_vec)], [0, maximum(A_vec)], 
#         color = "green", linestyle =:dash, labels = "Hill ≈ 0.5")
    return plt
end

"plot the hill function value"
function plot_hill(sol, _S, _D, _n)
    t_vec, σ_vec, A_vec = interpret_sol(sol)
    
    hill_vec = [duo_input_hill(x, y, _S, _D, _n) for (x, y) = zip(σ_vec, A_vec)]
    plt = plot(t_vec, hill_vec, title = "Hill function", legend = false)
    
    return plt
end

###############
# simulation
###############
"""
Simulate entrance

method: choose solver, available methods are "ssa" (Gillespie), "ode" and "sde"
parameters: including _v₀, _β, _KS, _rK (i.e. KD/KS), _n, _τ₁, _rτ (i.e. τ₂/τ₁), _η
max_t: maximum time span, default 2000.
stress_t: the time point that adds stress, default 200.
show_tc: show timecourse plot, default true
show_pp: show phase plane plot, default true
show_hill: show hill func. plot, default true
"""
function simu_all(_m; _v₀ = 0.01, _β = 100., _KS = 0.2, _rK = 1.,
        _n = 3., _τ₁ = 10., _rτ = 1., _η = .1, 
        max_t = 2000., stress_t = 200., plot_max_t = 2000., saveat = 1.0,
        method = "ssa", show_tc = true, show_pp = true, show_hill = true,
        quiet = false)
    tspan = (0., max_t)    # time course
    u₀ = [0, 0]  # initial state, σ A

    # parameters, v₀ β S D n τ₁ τ₂ η
    # S is initially set to 0 and subject to a step change
    p = [_v₀, _β, 1 / _KS, 1 / (_rK * _KS), _n, _τ₁, _τ₁ * _rτ, _η]
    
    # choose different methods
    if method == "ssa"
        sol = simu_ssa(_m, tspan, u₀, p, stress_t, _saveat = saveat)
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
        println("KD/KS = ", _rK, "; KS = ", _KS, "; τ₂/τ₁ = ", _rτ,
            "; β= ", _β, "; n = ", _n)
    end
    
    # plot
    if !quiet
        if show_tc
            display(plot_timecourse(sol, stress_t, max_t = plot_max_t))    # time course
        end
        if show_pp
            display(plot_phase_plane(sol))    # phase plane
        end
        if show_hill
            display(plot_hill(sol, 1 / _KS, 1 / (_rK * _KS), _n))
        end
    end
    
    # return arrays
    return sol
end

"Call back to add stress"
function S_step(step_time, step_value)
    # in solve, must use tstops to specify the break point
    condition(u, t, integrator) = (t == step_time)
    affect!(integrator) = integrator.p[3] += step_value    # S
    return DiscreteCallback(condition, affect!, save_positions = (true,true))
end

"simulate with Gillespie"
function simu_ssa(_m, tspan, u₀, p, stress_t; _saveat = 1.)
    _S = p[3]    # save S value
    p[3] = 0.    # initially set S = 0
    
    # Gillespie
    dprob = DiscreteProblem(_m, u₀, tspan, p)
    jprob = JumpProblem(_m, dprob, Direct(), save_positions = (false, false))
    jsol = solve(jprob, SSAStepper(),
        callback = S_step(stress_t, _S),
        tstops = [stress_t],
        saveat = _saveat)
    return jsol
end

"Simulate with ODE"
function simu_ode(_m, tspan, u₀, p, stress_t; _saveat = 1.)
    _S = p[3]    # save S value
    p[3] = 0.    # initially set S = 0
    
    oprob = ODEProblem(_m, u₀, tspan, p)
    sol = solve(oprob, Tsit5(), saveat = _saveat,
        callback = S_step(stress_t, _S),
        tstops = [stress_t])
    
    return sol
end

"In SDE, prevent blowup caused by neg. number"
function positive_domain()
    condition(u,t,integrator) = minimum(u) < 0.0
    function affect!(integrator)
        #integrator.u .= integrator.uprev
        integrator.u[integrator.u .< 0.0] .= 0.0
    end
    return DiscreteCallback(condition,affect!,save_positions = (false, false)) 
end

"Simulate with SDE"
function simu_sde(_m, tspan, u₀, p, stress_t; _saveat = 1.)
    _S = p[3]    # save S value
    p[3] = 0.    # initially set S = 0
    
    sprob = SDEProblem(_m, u₀, tspan, p,
        noise_scaling = (@variables η)[1])
    sol = solve(sprob, ImplicitEM(), 
        saveat = _saveat, 
        callback = CallbackSet(S_step(stress_t, _S), positive_domain()),
        tstops = [stress_t],
        maxiters = 100000000000,    # required for the "large maxiters needed" warning
        dt = 5e-3,
        adaptive = false,    # together with dt forces small step size
        force_dtmin = false)    # force continuation after "dt <= dtmin" warning
    
    return sol
end

##################
# fixed point analysis
##################
"""
Get the averaged vector field
"""
function vector_field(sol, stress_t, dt)
    t_vec, σ_vec, A_vec = interpret_sol(sol)
    dσdt = zeros(Float64, (maximum(σ_vec) + 1, maximum(A_vec) + 1))
    dAdt = zeros(Float64, (maximum(σ_vec) + 1, maximum(A_vec) + 1))
    passage = zeros(Int128, (maximum(σ_vec) + 1, maximum(A_vec) + 1))
    
    for i = 2:(length(t_vec) - 1)
        if t_vec[i] >= stress_t
            dσ = σ_vec[i + 1] - σ_vec[i - 1]
            dA = A_vec[i + 1] - A_vec[i - 1]
            # states start from 0 but the index of the array starts from 1
            dσdt[σ_vec[i] + 1, A_vec[i] + 1] += dσ/(2*dt)
            dAdt[σ_vec[i] + 1, A_vec[i] + 1] += dA/(2*dt)
            passage[σ_vec[i] + 1, A_vec[i] + 1] += 1
        end
    end
    
    # avoid divide 0 error
    pos_passage = deepcopy(passage)
    pos_passage[pos_passage .== 0] .= 1
    
    # normalize
    dσdt ./= pos_passage
    dAdt ./= pos_passage
    
    magnitude = sqrt.(dσdt.^2 .+ dAdt.^2)
    return dσdt, dAdt, magnitude, passage
end

"""
Vector field plot with quiver
please specify an interval or the plotting is extremely slow
and the vectors are clustered
"""
function plot_vf(dσdt, dAdt; scale = 1.0, interval = 5, adaptive = true)
    # if adaptive = true, interval is override
    σ_size = size(dσdt, 1)
    A_size = size(dσdt, 2)
    # calculate interval
    if adaptive
        # be careful with small grid
        interval = max(Int64(round(min(σ_size, A_size) / 10)), 1)
    end
    # make the matrix sparse
    dσdt = dσdt[1:interval:σ_size, 1:interval:A_size]
    dAdt = dAdt[1:interval:σ_size, 1:interval:A_size]
    # plot
    σ_grid = ((1:interval:σ_size) .- 1) * ones(1, Int64(ceil(A_size/interval)) )
    A_grid = ones(Int64(ceil(σ_size/interval)), 1) * ((1:interval:A_size) .- 1)'
    quiver(σ_grid[:], A_grid[:], quiver = (scale .* dσdt[:], scale .* dAdt[:]), 
        alpha = 0.5, color = "blue")
end

"""
find the fixed points
by the criterion: local minimum of vector magnitude
and local maximum of trajectory density
"""
function find_fp(sol, stress_t, dt, β; smooth_size = 3, neighbor_size = 5, thres_v = 2e-3, thres_d = 1e-4)
    # generate (magnitude of) vf, traj density
    dσdt, dAdt, magnitude, passage = vector_field(sol, stress_t, dt)
    
    # smooth
    ms = naive_smooth_2d(magnitude, smooth_size)
    ps = naive_smooth_2d(passage, smooth_size)

    # find local min/max
    v_min = local_min_2d(ms, neighbor_size)
    d_min = local_min_2d(-ps, neighbor_size)
    
    # find intersect with tolerance
    # also the density has to be large enough, and vf small
    n_fp = 0
    fp = zeros(max(size(d_min, 1), size(v_min, 1)), 2)    # initialize
    # max distance between density max & mag. min
    dist_tol = neighbor_size + 1
    # absolute value threshold
    #thres_v = 1e-2
    #thres_d = 1e-2
    len_traj = length(sol.t) - Int64(round(stress_t / dt))
    for i = 1:size(v_min, 1)
        if magnitude[v_min[i, :]...] > β * thres_v ||
            passage[v_min[i, :]...] < len_traj * thres_d
            continue
        end
        for j = 1:size(d_min, 1)
            dist = sqrt(sum((d_min[j, :] .- v_min[i, :]).^2))
            if dist < dist_tol
                n_fp += 1
                fp[n_fp, :] .= v_min[i, :]
                break
            end
        end
    end
    
    fp = fp[1:n_fp, :]
    fp .-= 1    # convert index to # moleclues
    return fp, v_min, d_min
end

"""
marking the fixed points on the vector field
"""
function plot_vf_w_fp(dσdt, dAdt, fp; scale = 1.0, interval = 5, adaptive = true)
    plt = plot_vf(dσdt, dAdt, scale = scale, interval = interval, adaptive = adaptive)

    scatter!(plt, fp[:, 1], fp[:, 2], color = :green, markersize = 10, 
        markerstrokewidth = 0, label = "fixed points", legend = :topright)
end

function naive_smooth_2d(mat, smooth_size = 5)
    new_mat = zeros(size(mat))
    half_smooth_size = Int64(floor(smooth_size/2))
    for i = 1:size(mat, 1)
        for j = 1:size(mat, 2)
            x_end = min(i + half_smooth_size, size(mat, 1))
            x_start = max(i + 1 - (smooth_size - half_smooth_size), 1)
            y_end = min(j + half_smooth_size, size(mat, 2))
            y_start = max(j + 1 - (smooth_size - half_smooth_size), 1)
            new_mat[i, j] = mean(mat[x_start:x_end, y_start:y_end])
        end
    end
    return new_mat
end

"""
find local minimum
the result is given by a col of x and a col of y
"""
function local_min_2d(mat, neighbor_size = 1)
    δ = 1e-6    # threshold for minimum
    n_min = 0
    x_idx = zeros(Int64, length(mat[:]))
    y_idx = zeros(Int64, length(mat[:]))
    for i = 1:size(mat)[1]
        for j = 1:size(mat)[2]
            x_start = max(1, i - neighbor_size)
            x_end = min(size(mat)[1], i + neighbor_size)
            y_start = max(1, j - neighbor_size)
            y_end = min(size(mat)[2], j + neighbor_size)
            sorted = sort(mat[x_start:x_end, y_start:y_end][:])
            # strictly greater than
            if sorted[1] == mat[i, j] && sorted[2] - mat[i, j] > δ * abs(mat[i, j])
                n_min += 1
                x_idx[n_min] = i
                y_idx[n_min] = j
            end
        end
    end
    x_idx = x_idx[1:n_min]    # truncate
    y_idx = y_idx[1:n_min]
    return [x_idx y_idx]
end

####################
# Classification
####################
function classify_by_timetraj(sol, stress_t, dt, β, n; smooth_size = 3, neighbor_size = 5, 
        thres_v = 2e-3, thres_d = 1e-4, thres_f = 1e-7, fluc_mult = 3, 
        show_vf_plot = false, show_v_heatmap = false, show_p_heatmap = false, quiet = true)
    # find fixed points
    fp, v_min, p_max = find_fp(sol, stress_t, dt, β; smooth_size = smooth_size, 
        neighbor_size = neighbor_size, thres_v = thres_v, thres_d = thres_d)
    
    # theoretical sqrt(Var(x)) / <x>
    fluc_level = sqrt(β / n)
    #fluc_mult = 1    # beyond fluc_level * fluc_mult is considered a pulse

    # condition 1: # of fp
    n_fp = size(fp, 1)

    # condition 2: # of fp below or above the noise level
    is_small_fp = [fp[i, 1] < fluc_mult * fluc_level && 
        fp[i, 2] < fluc_mult * fluc_level for i = 1:size(fp, 1)]
    n_small_fp = sum(is_small_fp)
    is_large_fp = [fp[i, 1] >= fluc_mult * fluc_level && 
        fp[i, 2] >= fluc_mult * fluc_level for i = 1:size(fp, 1)]
    n_large_fp = sum(is_large_fp)

    # condition 3: flow passes the noise level?
    len_traj = length(sol.t) - Int64(round(stress_t / dt))
    dσdt, dAdt, magnitude, passage = vector_field(sol, 200.0, 1.0)
    fluc_level_int = Int64(round(fluc_level))
    if size(magnitude, 2) <= fluc_mult * fluc_level_int
        reverse_flow = 0.0    # traj never reched 3 * fluc_level
    else
        reverse_flow = mean(magnitude[
                1:min(size(magnitude, 1), fluc_mult * fluc_level_int), fluc_mult * fluc_level_int])
    end
    has_reverse_flow = reverse_flow > len_traj * thres_f
    if size(magnitude, 1) <= fluc_mult * fluc_level_int
        forward_flow = 0.0    # traj never reched 3 * fluc_level
    else
        forward_flow = mean(magnitude[
                fluc_mult * fluc_level_int, 1:min(size(magnitude, 2), fluc_mult * fluc_level_int)])
    end
    has_forward_flow = forward_flow > len_traj * thres_f
    
    # decide
    if n_fp == 0
        if has_reverse_flow && has_forward_flow
            regime = :oscillation
        else
            # noisy activation or no expression detected
            # this is a bit sloppy
            if size(magnitude, 1) < fluc_mult * fluc_level && size(magnitude, 2) < fluc_mult * fluc_level
                regime = :no_expression
            else
                regime = :homo_activation
            end
#             regime = :undefined
        end
    elseif n_fp == 1
        if n_small_fp == 1
            if has_reverse_flow || has_forward_flow
                regime = :stochastic_pulsing
            else
                regime = :no_expression
            end
        else
            if has_reverse_flow && has_forward_flow
                regime = :stochastic_anti_pulsing
            else
                regime = :homo_activation
            end
        end
    elseif n_fp == 2
        if n_small_fp == 1
            if has_reverse_flow && has_forward_flow
                regime = :stochastic_switching
            else
                regime = :het_activation
            end
        else
            regime = :undefined
        end
    else
        regime = :undefined
    end
    # report
    if !quiet
        println("# fp: ", n_fp, ", # fp below noise level: ", n_small_fp,
            ", # fp above noise level: ", n_large_fp, ", has reverse flow: ", has_reverse_flow,
            ", has forward flow: ", has_forward_flow)
        println("The dynamical behaviour is: ", regime)
    end
    
    # plot
    if !quiet
        if(show_vf_plot)
            display(plot_vf_w_fp(dσdt, dAdt, fp))
        end
        if(show_v_heatmap)
            display(heatmap_w_extrema(magnitude, v_min, fluc_mult * fluc_level))
        end
        if(show_p_heatmap)
            display(heatmap_w_extrema(passage, p_max, fluc_mult * fluc_level))
        end
    end
    
    return regime
end

function heatmap_w_extrema(magnitude, min_idx, noise_level)
    # min_idx is the index of extrema, the coord is min_idx - 1
    p_heatmap = heatmap(1:size(magnitude, 1), 1:size(magnitude, 2), magnitude')
    plt = scatter!(p_heatmap, min_idx[:, 1], min_idx[:, 2],
        color = :lightblue, legend = false, markersize = 5)    # for clearity hide the legend
    plt = plot!(plt, [noise_level], seriestype = "vline", color = "red", 
        linestyle = :dash, legend = false)
    plt = plot!(plt, [noise_level], seriestype = "hline", color = "red", 
        linestyle = :dash, legend = false)
end