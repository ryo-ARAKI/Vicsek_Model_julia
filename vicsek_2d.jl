#=
Vicsek model in 2-D periodic region
=#

"""
Module for parameters and variables
"""
module  mod_param_var
    struct Parameters
        N::Int64  # Number of particles
        R_0::Float64  # Neighbour region threshold
        η::Float64  # Coefficient of white noise
        t_step::Int64 # Total iteration steps
        v0::Float64  # Velocity of particle (same for all particles)
    end

    mutable struct Variables
        r::Array{Float64, 2}    # Position of particles
        θ::Array{Float64, 1}    # Angle of particles
        n_label::Array{Bool, 2} # Neighbourhood label
        ψ::Array{Float64, 1}    # Neighbour alignment
        ξ::Array{Float64, 1}    # White noise for perturbation
        r_new::Array{Float64, 2}    # New position of particles
        θ_new::Array{Float64, 1}    # New angle of particles
    end

    mutable struct StatisticalValues
        θ_ave::Float64  # Average of θ: Angle of particles
        θ_var::Float64  # Variance of θ: Angle of particles
    end
end  # module mod_param_var


"""
Module for Vicsek model time integration
"""
module mod_vicsek_model
    """
    Initialise position of particles with [0,1]×[0,1] random number.
    """
    function set_initial_condition(param,var)
        var.r = rand(Float64, param.N, 2)
        θ_ = rand(Float64, param.N)
        var.θ = 2π*θ_ .- π
    end

    """
    Calculate distance between two particlesi and j,
    if distance is below threshold R_0, n_label (neighbour_label) have true.
    """
    function set_neighbour_list(param,var)
        #=
        この処理がN^2を要する
        要高速化
        領域をR_0×R_0の矩形に分割し，隣接矩形でのみ検索する
        MPI
        =#
        for i=1:param.N
            for j=1:param.N
                dist = sqrt((var.r[i,1]-var.r[j,1])^2+(var.r[i,2]-var.r[j,2])^2)
                if dist <= param.R_0
                    var.n_label[i,j] = true
                else
                    var.n_label[i,j] = false
                end
            end
        end
    end

    """
    Calculate one particle's neighbour orientation
    ψ will have [-π,π] value defined as
        ψ = Arg[Σ_j n_ij θ_j].
    """
    function set_neighbour_orientation(param,var)
        var.ψ = zeros(param.N)
        #=
        この処理がN^2を要する
        要高速化
        領域をR_0×R_0の矩形に分割し，隣接矩形でのみ検索する
        MPI
        =#
        for i=1:param.N
            for j=1:param.N
                var.ψ[i] = var.ψ[i] + var.n_label[i,j]*var.θ[j]
            end
        end
    end

    """
    Calculate white noise array ξ.
    """
    function set_white_noise(param,var)
        var.ξ = rand(Float64, param.N)
    end

    """
    Calculate θ at time t+Δt: θ_new
        θ_new = ψ + ηξ.
    """
    function set_new_θ(param,var)
        var.θ_new = var.ψ .+ param.η*var.ξ
        # Try to ensure θ_new have [-π,π] value
        #=
        modなどで代替&簡略化できないか?
        =#
        for i=1:param.N
            if var.θ_new[i] > π
                var.θ_new[i] -= π
            elseif var.θ_new[i] < -π
                var.θ_new[i] += π
            end
        end
    end

    """
    Calculate r at time t+Δt: r_new
        r_new = (x_new,y_new), r = (x,y)
        x_new = x + Δt*v_0*cos(θ_new)
        y_new = y + Δt*v_0*sin(θ_new)
        Δt = v_0 = 1
    """
    function set_new_r(param,var)
        #=
        forループ使わずに書きたい
        =#
        for i=1:param.N
            var.r_new[i,1] = var.r[i,1] + param.v0*cos(var.θ_new[i])
            var.r_new[i,2] = var.r[i,2] + param.v0*sin(var.θ_new[i])
        end
    end

    """
    Ensure periodic boundary condition
    """
    function set_periodic_bc(param,var)
        for i=1:param.N
            for j=1:2
                if var.r_new[i,j] > 1.0
                    var.r_new[i,j] -= 1.0
                elseif var.r_new[i,j] < 0
                    var.r_new[i,j] += 1.0
                end
            end
        end
    end

    """
    Update r & θ
    """
    function set_new_rθ(param,var)
        var.r = var.r_new
        var.θ = var.θ_new
    end
end  # module mod_vicsek_model


"""
Module for analysing Vicsek model
"""
module mod_analysis
    """
    Calculate average and variance of direction of particles
    """
    function calc_ave_var_θ(param,θ)
        θ_ave = 0.0
        for i=1:param.N
            θ_ave += θ[i]
        end
        θ_ave = θ_ave / param.N
        θ_var = 0.0
        for i=1:param.N
            θ_var += (θ[i]-θ_ave) ^ 2
        end
        θ_var = θ_var / param.N
        return θ_ave, θ_var
    end
end  # module mod_analysis


## Declare modules
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
calc_ave_var_θ


## Set parameter
N = 10
R_0 = 0.05
η = 0.05
t_step = 10
v0 = 0.02
param_ = mod_param_var.Parameters(N,R_0,η,t_step,v0)

## Set variables
r = Array{Float64}(undef, param_.N, 2)
θ = Array{Float64}(undef, param_.N)
n_label = BitArray(undef, param_.N, param_.N)
ψ = Array{Float64}(undef, param_.N)
ξ = Array{Float64}(undef, param_.N)
r_new = Array{Float64}(undef, param_.N, 2)
θ_new = Array{Float64}(undef, param_.N)
var_ = mod_param_var.Variables(r,θ,n_label,ψ,ξ,r_new,θ_new)

## Set statistical values
θ_ave = 0.0
θ_var = 0.0
sta_ = mod_param_var.StatisticalValues(θ_ave,θ_var)


## Main
set_initial_condition(param_,var_)
println(var_.r)  # Position
println(var_.θ)  # Direction

for t=1:param_.t_step
    set_neighbour_list(param_,var_)
    # println(var_.n_label[1,:])  # Neighbour list of particle 1
    set_neighbour_orientation(param_,var_)
    # println(var_.ψ)  # Neighbour orientation
    set_white_noise(param_,var_)
    # println(var_.ξ)  # white noise
    set_new_θ(param_,var_)
    # println(var_.θ_new)  # Updated direction
    set_new_r(param_,var_)
    # println(var_.r_new)  # Updated position
    set_periodic_bc(param_,var_)
    # println(var_.r_new)  # Periodic b.c.-ensured updated position
    set_new_rθ(param_,var_)
    # println(var_.r)  # Updated position
    # println(var_.θ)  # Updated direction
    sta_.θ_ave,sta_.θ_var = calc_ave_var_θ(param_,var_.θ)
    println("θ ave:",sta_.θ_ave," ,θ var:",sta_.θ_var)
    #=
    アニメーション作成
    =#
end
