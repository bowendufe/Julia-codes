using QuantEcon, Plots, Interpolations, Random, Parameters, Statistics
# Close plots


# Function that sets parameters
function parameters(
                    # Pass all input after the semicolon, position does not matter.
                    ;
                    nkk::Integer   = 100,   # Points in asset grid.
                    nzz::Integer   = 5,    # Points in productivity shock grid
                    J::Integer     = 71,    # number of age groups
                    Jr::Integer    = 41,    # period of retirement
                    α::Float64     = 0.36,  # capital share
                    β::Float64     = 0.96,  # discount factor
                    δ::Float64     = 0.08,  # depreciation rate
                    σ::Float64     = 2.0,   # risk aversion
                    ω::Float64     = 0.5,   # replacement rate
                    n::Float64     = 0.01,  # growth rate of new cohort
                    k_lb::Float64  = 0.0,   # lower bound of asset grid
                    k_ub::Float64  = 100.0, # upper bound of asset grid
                    ρ_z::Float64   = 0.9,   # persistence of productivity shock
                    σ_z::Float64   = 0.2,   # variance of error term of AR1
                    spread::Integer = 2)    # bounds of prodcutivity grid in terms of standard deviations

    # Grid for capital holdings
    a_grid = collect(range(k_lb,k_ub,length=nkk))

    # Construct grid for productivity
    DiscretizedMC = QuantEcon.tauchen(nzz,ρ_z,σ_z,0.0,spread)
    aux           = QuantEcon.stationary_distributions(DiscretizedMC)
    Πstat         = aux[1]
    log𝐳          = DiscretizedMC.state_values
    Π             = DiscretizedMC.p
    𝐳             = exp.(log𝐳)
    mean𝐳         = sum(𝐳.*Πstat)
    z_grid        = 𝐳./mean𝐳;
    #avgz         = sum(𝐳.*Π_stat) # It is equal to one, okay.

    # Load survival probabilities and compute population fractions
    #aux2     = readdlm("LifeTables.txt",Float64)   # Read surv. prob. table
    #s        = aux2[:,1]                           # Store survival probabilities
    N        = Array{Float64}(undef,J)             # Pop. size by age group
    N[1] = 1.0
    for j in 2:J
        N[j] = N[j-1]/(1+n)
    end
    ψ = N ./ sum(N)                                # Pop. shares by group

    # Create vector of efficiency units of labor for each (j,z) pair
    λ = zeros(J)                               # Avg. life-cycle income profile
    for j in 1:Jr-1
        λ[j] = 0.195 + 0.107*j - 0.00213*j^2     # Fill it with provided polynomial
    end
    e = zeros(nzz,J)                             # Efficiency units of labor
    for j in 1:Jr-1
        for z in 1:nzz
            e[z,j] = z_grid[z] * λ[j]
        end
    end

    # Labour supply
    L = sum(ψ.*λ)                                # Scalar prod. of pop. shares and avg. inc.

    # Payroll tax
    θ = ω*sum(ψ[Jr:end])/sum(ψ[1:Jr-1])

    # Assign types to relevant parameters (thanks, Andrej)
    return (nkk=nkk, nzz=nzz,J=J, Jr=Jr, a_grid=a_grid,z_grid= z_grid, α=α,β= β,δ= δ, σ=σ,ω= ω,n= n,Π= Π,Πstat= Πstat,DiscretizedMC= DiscretizedMC,
            ψ=ψ,e= e,L= L,θ= θ,λ= λ)
end
function CapitalWagesPensions(param::parstruct,r::Float64)
    @unpack α, δ, L, θ, ψ, Jr, J = param
    K =  ((r+δ)/(α))^(1/(α-1))*L
    w = (1 - α) * (α / (r + δ))^(α / (1 - α))
    b          = zeros(J)
    b[Jr:end] .= θ*w*L/(sum(ψ[Jr:end]))
    return K, w, b
end
function HH_EGM(
    param,
    r,
    T,
    w)
    @unpack a_grid, z_grid, nkk, nzz, J, e, σ, β, Π, θ  = param
    g = zeros(nzz,nkk,J)
    # Compute policy function from J-1:-1:1 (nosavings in last period J)
    for j in J-1:-1:1
        a_star = zeros(nzz,nkk)
        for k in 1:nkk
            for z in 1:nzz
                # Compute income (except asset income) this period and next period
                y  = (1-θ)*e[z,j]*w+T'                     # 1×1
                y′ = (1-θ)*e[:,j+1]*w  .+T             # z×1

                # Calculate a consistent with a′ and g(z′,a′,j+1),
                a_star[z,k] = ((β*(1+r)*(Π[z,:]'*((y′ .+ (1+r)*a_grid[k] - g[:,k,j+1]).^(-σ))))^(-1/σ) - y + a_grid[k])/(1+r)
            end
        end

        # interpolate to obtain policy function
        for k in 1:nkk
            for z in 1:nzz
                nodes = (a_star[z,:],)
                itp  = interpolate(nodes,a_grid,Gridded(Linear()))
                etpf = extrapolate(itp,Line())
                g[z,k,j] = etpf(a_grid[k])

                # corner solutions
                if g[z,k,j] < 0
                   g[z,k,j] = 0.0
                end
            end
        end
    end
    return g
end
param = parameters()
g=HH_EGM(param,0.03,1.0,1.0)
g[:,:,35]'
h=parameters()
function find_a(V::Array{Float64},i_t::Int64,i_z::Int64,h,a::Float64,e_val::Float64,r::Float64,w::Float64,T::Float64)
    function opt_a(aprime)
        c=a*(1+r)+(1-h.θ)*e_val*w+T-aprime
        interpV=zeros(h.nzz)

        for zp=1:h.nzz
            itp=interpolate((h.a_grid,),vec(V[:,zp,i_t+1]),Gridded(Linear()))
            extp=extrapolate(itp,Linear())
            interpV[zp]=extp[aprime]
        end

        exp_valre=transpose(h.Π[i_z,:])*interpV
        exp_val=exp_valre[1]

        utility_flow=c^(1-h.σ)/(1-h.σ)
        Vnew=(utility_flow+h.β*exp_val)*(-1.0)

        return Vnew
    end

    results=optimize(opt_a,0.0,a*(1+r)+(1-h.θ)*e_val*w+T)
    aprime=Optim.minimizer(results)
    V=(-1.0)*Optim.minimum(results)
    c=a*(1+r)+(1-h.θ)*e_val*w+T-aprime
    return aprime,V,c
end

function hhdecision(h)
    @unpack a_grid, z_grid, nkk, nzz, J, e, σ, β, Π, θ  = h
    r=0.03
    w=1.0
    T=1.0

    # calculate L

    # measure old

    V=zeros(nkk,nzz,J+1) # value for the agents by assets, earnings shock, and age
    aprime=similar(V) # asset choice
    cons=similar(V)

    for i_t=1:J
        for (i_z,z) in enumerate(z_grid)
            for (i_a,a) in enumerate(a_grid)
                aprime[i_a,i_z,J+1-i_t],V[i_a,i_z,J+1-i_t],cons[i_a,i_z,J+1-i_t]=find_a(V,i_t,i_z,h,a,e[i_z,i_t],r,w,T)
            end
        end
    end

    return V,aprime,cons
end
v,apol,c=hhdecision(h)
