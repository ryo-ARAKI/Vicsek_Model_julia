#=
Vicsek model in 2-D periodic region
=#

"""
Module for parameters and variables
"""
module  mod_param_var

struct Parameters
    ρ::Float64  # Number of particles per unit area
    η::Float64  # Coefficient of white noise
    v0::Float64  # Velocity of particle (same for all particles)
end

struct Constants
    xrange::Float64  # Actual computation domain size
    N::Int64  # Actual number of particle
    t_step::Int64  # Total time steps
end

mutable struct Variables
    itr::Int64    # Number of iteration
    r::Array{Float64, 2}    # Position of particles
    θ::Array{Float64, 1}    # Angle of particles
    n_label::Array{Bool, 2} # Neighbourhood label
    ψ::Array{Float64, 1}    # Neighbour alignment
    ξ::Array{Float64, 1}    # White noise for perturbation
    r_new::Array{Float64, 2}    # New position of particles
    θ_new::Array{Float64, 1}    # New angle of particles
    n_sum::Array{Int64, 1}      # Number of nearby particles
end

mutable struct StatisticalValues
    φ::Array{Float64, 2}  # Polar order parameter
    φ_::Array{Float64, 1}  # Polar order parameter
end

end  # module mod_param_var


"""
Module for Vicsek model time integration
"""
module mod_vicsek_model
using Statistics

"""
Initialise position of particles with [0,1]×[0,1] random number.
"""
function set_initial_condition(consts, var)
    tmp = rand(Float64, consts.N, 2)
    var.r = tmp .* consts.xrange # [0, 1] -> [0, xrange]
    tmp = rand(Float64, consts.N)
    var.θ = 2π*tmp .- π  # [0,1] -> [-π,π]
end

"""
Calculate distance between two particles i and j,
if distance is below threshold R_0, n_label (neighbour_label) have true.
n_label of myself(tr(n_label)) is 1
Also, calculate number of neighbour particles and store in var.n_sum
"""
function set_neighbour_list(consts, var)
    #=
    この処理がN^2を要する
    要高速化
    領域をR_0×R_0の矩形に分割し，隣接矩形でのみ検索する
    MPI
    =#
    for i=1:consts.N
        n_col = 0  # Number of colums of n_label[i,:]
        for j=1:consts.N
            # Calculate distance between two particles
            dx = abs(var.r[i,1] - var.r[j,1])
            dx = min(dx, consts.xrange - dx)
            dy = abs(var.r[i,2] - var.r[j,2])
            dy = min(dy, consts.xrange - dy)
            dist = hypot(dx, dy)
            if dist <= 1.0
                var.n_label[i,j] = true
                n_col += 1
            else
                var.n_label[i,j] = false
            end
        end
        var.n_sum[i] = n_col
    end
    # println("average % of num. of neighbour=", mean(var.n_sum)/consts.N * 100.0)
end

"""
Calculate one particle's neighbour orientation
ψ will have [-π,π] value defined as
    ψ = Arg[Σ_j n_ij θ_j].
"""
function set_neighbour_orientation(consts, var)
    #=
    この処理がN^2を要する
    要高速化
    領域をR_0×R_0の矩形に分割し，隣接矩形でのみ検索する
    MPI
    =#
    for i=1:consts.N
        tmpx = tmpy = 0.0
        for j=1:consts.N
            tmpx += var.n_label[i,j] * cos(var.θ[j])
            tmpy += var.n_label[i,j] * sin(var.θ[j])
        end
        var.ψ[i] = atan(tmpy, tmpx)
    end
end

"""
Calculate white noise array ξ.
"""
function set_white_noise(consts, var)
    ξ_ = rand(Float64, consts.N)
    var.ξ = 2π .* ξ_  .- π # [0,1] -> [-π,π]
end

"""
Calculate θ at time t+Δt: θ_new
    θ_new = ψ + ηξ.
"""
function set_new_θ(param, consts, var)
    for i=1:consts.N
        var.θ_new[i] = var.ψ[i] + param.η * var.ξ[i]  # ensure θ_new ∈ [-π,π]
    end
end

"""
Calculate r at time t+Δt: r_new
    r_new = (x_new,y_new), r = (x,y)
    x_new = x + Δt*v_0*cos(θ_new)
    y_new = y + Δt*v_0*sin(θ_new)
    Δt = v_0 = 1
"""
function set_new_r(param, consts, var)
    for i=1:consts.N
        var.r_new[i,1] = var.r[i,1] + param.v0*cos(var.θ_new[i])
        var.r_new[i,2] = var.r[i,2] + param.v0*sin(var.θ_new[i])
    end
end

"""
Ensure periodic boundary condition
"""
function set_periodic_bc(consts, var)
    for i=1:consts.N
        for j=1:2
            if var.r_new[i,j] > consts.xrange
                var.r_new[i,j] -= consts.xrange
            elseif var.r_new[i,j] < 0
                var.r_new[i,j] += consts.xrange
            end
        end
    end
end

"""
Update r & θ
"""
function set_new_rθ(var)
    var.r = var.r_new
    var.θ = var.θ_new
end
end  # module mod_vicsek_model


"""
Module for analysing Vicsek model
"""
module mod_analysis
"""
Calculate direction parameter φ
    φ = 1/N Σ_{i=1}^N s_i^t
    s_i^t is a unit direction vector, implemented as (cosθ,sinθ)
"""
function calc_φ(consts,θ)
    tmp_c = 0.0
    tmp_s = 0.0
    for i=1:consts.N
        tmp_c += cos(θ[i])
        tmp_s += sin(θ[i])
    end
    tmp_c = tmp_c/consts.N
    tmp_s = tmp_s/consts.N
    tmp = sqrt(tmp_c^2 + tmp_s^2)
    return [tmp_c, tmp_s], tmp
end

end  # module mod_analysis


"""
Module for dat, image and movie generation
"""
module mod_output

using Plots
"""
Output snapshot image of particle distribution and direction
"""
function plot_scatter_φ(param, consts, var, stat, flag_out)
    u = Array{Float64}(undef, consts.N)
    v = Array{Float64}(undef, consts.N)
    for i=1:consts.N
        u[i] = 0.05 * cos(var.θ[i])
        v[i] = 0.05 * sin(var.θ[i])
    end
    p1 = quiver(  # Vector field
        var.r[:,1], var.r[:,2],
        quiver=(u[:], v[:]),
        aspect_ratio = 1,
        xlims = (0.0, consts.xrange),
        ylims = (0.0, consts.xrange),
        xaxis=nothing,
        yaxis=nothing,
        color=1
    )
    p1! = scatter!(  # Position of particles
        var.r[:,1],var.r[:,2],
        markerstrokewidth = 0,
        color=1  # Same color as quiver
    )
    #=
    quiverのベクトルのスタイルを変えたい(ベクトルのノルムに合わせて幅を変えたい)
    https://discourse.julialang.org/t/plots-jl-arrows-style-in-quiver/13659/2
    を見ると未実装らしい??
    =#
    p2 = plot(
        stat.φ_[1:var.itr],
        xlims = (0, consts.t_step),
        ylims = (0, 1.1),
        xaxis = ("Time step"),
        yaxis = ("Orientation parameter"),
        linewidth = 2)
    plot(p1,p2,size=(1260,480))
    if flag_out == true
        str_t = lpad(string(var.itr), 5, "0")  # iteration number in 5 digit, left-padded string
        str_ρ = lpad(string(param.ρ ), 3, "0")
        str_η = lpad(string(param.η), 3, "0")
        png("img/vicsek_ρ=$(str_ρ)_eta=$(str_η)_$(str_t).png")
    end
end

"""
"""
function make_gif(param, anim)
    str_ρ = lpad(string(param.ρ ), 3, "0")
    str_η = lpad(string(param.η), 3, "0")
    gif(
        anim,
        "img/vicsek_ρ=$(str_ρ)_eta=$(str_η).gif",
        fps=10)
end

function plot_φ(param, consts, var, stat)
    plot(
        stat.φ_[1:var.itr],
        xlims = (0, consts.t_step),
        ylims = (0, 1.1),
        xaxis = ("Time step"),
        yaxis = ("Orientation parameter"),
        linewidth = 2)
    xaxis!("Time step")
    yaxis!("Modulo of order parameter")
    str_ρ = lpad(string(param.ρ ), 3, "0")
    str_η = lpad(string(param.η), 3, "0")
    str_t = lpad(string(consts.t_step), 4, "0")
    png("img/phi_ρ=$(str_ρ)_eta=$(str_η)_$(str_t)step.png")
end

"""
Plot time-averaged φ_ versus noise amplitude η
"""
function plot_η_φ()
    plot(
        η[1:15],
        φ[1:15],
        marker= (
            :circle,  # shape of marker
            8,  # size of marker
            0  # transparency of marker
            # stroke(0,:white)  # stroke of marker
            ),
        linestyle = :solid,
        xaxis = ("Noise amplitude eta"),
        yaxis = ("Time-averaged orientation parameter varphi"),
        linewidth = 2)
    png("img/eta_phi.png")
end

end  # module mod_output


## Declare modules
using ProgressMeter
using Plots
gr(
    legend = false  # Default setting for all figures
)
using .mod_param_var  # Define parameters and variables
import .mod_vicsek_model:  # Definde time-integration of vicsek model
    set_initial_condition,
    set_neighbour_list,
    set_neighbour_orientation,
    set_white_noise,
    set_new_θ,
    set_new_r,
    set_periodic_bc,
    set_new_rθ
import .mod_analysis:  # Define functions for analysis
    calc_φ
import .mod_output:  # Define functions for output data
    plot_scatter_φ,
    make_gif,
    plot_φ


## Set parameter
### Control parameters
ρ = 100.0  # Number of particles per unit area
η = 0.4  # Noise amplitude
v0 = 0.05  # Velocity of particles
param_ = mod_param_var.Parameters(ρ, η, v0)

### Other constants (defined by parameters)
xrange = 1.0  # Actual computation domain size
N = ρ * xrange^2  # Actual number of particle
t_step = 200
const_ = mod_param_var.Constants(xrange, N, t_step)

## Set variables
itr = 1
r = Array{Float64}(undef, const_.N, 2)
θ = Array{Float64}(undef, const_.N)
n_label = BitArray(undef, const_.N, const_.N)
ψ = Array{Float64}(undef, const_.N)
ξ = Array{Float64}(undef, const_.N)
r_new = Array{Float64}(undef, const_.N, 2)
θ_new = Array{Float64}(undef, const_.N)
n_sum = Array{Int64}(undef, const_.N)
var_ = mod_param_var.Variables(
    itr,
    r, θ,
    n_label, ψ, ξ,
    r_new, θ_new, n_sum
)

## Set statistical values
φ = Array{Float64}(undef, const_.t_step, 2)
φ_ = Array{Float64}(undef, const_.t_step)
sta_ = mod_param_var.StatisticalValues(φ,φ_)


## Main
set_initial_condition(const_, var_)

progress = Progress(const_.t_step)
# for var_.itr=1:const_.t_step
anim = @animate for var_.itr=1:const_.t_step
    set_neighbour_list(const_, var_)
    set_neighbour_orientation(const_, var_)
    set_white_noise(const_, var_)
    set_new_θ(param_, const_, var_)
    set_new_r(param_, const_, var_)
    set_periodic_bc(const_, var_)
    set_new_rθ(var_)
    sta_.φ[var_.itr,:], sta_.φ_[var_.itr] = calc_φ(const_, var_.θ)
    # println("itr=",var_.itr, " φ[1]=", sta_.φ[var_.itr,1], " φ[2]=",sta_.φ[var_.itr,2], " φ_=",sta_.φ_[var_.itr])
    plot_scatter_φ(param_, const_, var_, sta_, false)
    next!(progress)
end

make_gif(param_, anim)
plot_φ(param_, const_, var_, sta_)
println("")
println("time averaged φ_=",sum(sta_.φ_/const_.t_step))
