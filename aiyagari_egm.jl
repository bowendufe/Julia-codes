using LinearAlgebra
using Parameters
using QuantEcon
using Plots
using BenchmarkTools
using Interpolations
using Plots

function Household(;β::Float64=0.96,
                    γ::Float64=2.0,
    #y_chain::MarkovChain{Float64,Matrix{Float64},Vector{Float64}}=MarkovChain([0.5 0.5; 0.04 0.96], [0.25; 1.0]),
                   r::Float64=0.04,
                   amax::Float64=25.0,
                   Na::Int64=100)

    # Set up asset grid
    agrid = range(0.0,amax,length=Na)

    # Parameters of income grid
    #y_grid = y_chain.state_values
    y_grid=[1.0 0.2]'
    Pi=[0.93 0.07;0.5 0.5]
    Ny = length(y_grid)

    # Combined grids
    ay_grid = kron(agrid, ones(1,Ny))
    ayi_grid = kron(y_grid',ones(Na,1))

    # Set up initial guess for value function


    # Corresponding initial policy function is all zeros
    ap = zeros(Na, Ny)
    Ec = zeros(Na,Ny)
    return (β=β, Pi=Pi,γ=γ, r=r, amax=amax, Na=Na, agrid=agrid,
            y_grid=y_grid, ay_grid=ay_grid, ayi_grid=ayi_grid, Ny=Ny, ap=ap,Ec=Ec)

end


function interpEGM(pol::AbstractArray,
                grid::AbstractArray,
                x::T,
                na::Integer) where{T <: Real}
    np = searchsortedlast(pol,x)

    ##Adjust indices if assets fall out of bounds
    (np > 0 && np < na) ? np = np :
        (np == na) ? np = na-1 :
            np = 1
    #@show np
    ap_l,ap_h = pol[np],pol[np+1]
    a_l,a_h = grid[np], grid[np+1]
    ap = a_l + (a_h-a_l)/(ap_h-ap_l)*(x-ap_l)

    above =  ap > 0.0
    return above*ap,np
end
function solve_egm(h,g)
    @unpack β,r,Pi,γ,Ny,Na,agrid,y_grid,ay_grid=h
    a_star=zeros(Ny,Na)
    c_star=similar(a_star)
    #g=ay_grid
    c=similar(a_star)
    #cpol=similar(Ec)
    #cp=similar(Ec)
    for k in 1:Na
        for z in 1:Ny
            # Compute income (except asset income) this period and next period
            y  = y_grid[z]                     # 1×1
            y′ = y_grid[:]          # z×1

            # Calculate a consistent with a′ and g(z′,a′,j+1),
            a_star[z,k] = ((β*(1+r)*(Pi[z,:]'*((y′ .+ (1+r)*agrid[k] - g[:,k]).^(-γ))))^(-1/γ) - y + agrid[k])/(1+r)
            c_star[z,k] = (1+r)*a_star[z,k]+y-agrid[k]
        end
    end

    # interpolate to obtain policy function


    #for i=1:h.Ny
    #    for j=1:h.Na
    #        app[j,i]=interpEGM(apol[:,i],h.a_grid,h.a_grid[j,1],h.Na)[1]
    #        itp=LinearInterpolation(apol[:,i],h.a_grid;extrapolation_bc=Line())
    #        app[j,i]=itp(h.a_grid[j,1])
    #    end
    #end
    #cpol=-app.+ h.ayi_grid.+(1+r)*h.ay_grid
    return a_star,c_star
end

function update(h,apol,cpol)
    @unpack β,r,Pi,γ,Ny,Na,agrid,y_grid,ay_grid=h
    a_star=copy(apol)
    c_star=copy(cpol)
    c=zeros(Ny,Na)
    g=zeros(Ny,Na)
    for k in 1:Na
        for z in 1:Ny
            nodes = a_star[z,:]
            cnodes= c_star[z,:]
            g[z,k]=interpEGM(nodes,agrid,agrid[k,1],Na)[1]
            #itp  = interpolate(nodes,agrid,Gridded(Linear()))
            #etpf = extrapolate(itp,Line())
            #g[z,k,j] = etpf(agrid[k])
            c[z,k]=interpEGM(nodes,cnodes,agrid[k,1],Na)[1]
            # corner solutions
            if g[z,k] < 0
                g[z,k] = 0.0
               if c[z,k] < 0
                   c[z,k]=0.0
               end
            end
        end
    end
    return g
end



function solve_bellman(h,tol=1e-5)
    @unpack Na,Ny,ay_grid=h
    #pol=copy(ay_grid)
    #anew=zeros(Na,Ny)
    pol=ay_grid'
    a=zeros(Ny,Na)
    for i = 1:1000
        anew = copy(a)
        a ,cpp= solve_egm(h,pol)
        pol=update(h,a,cpp)
        #app=update_ap(h,a)



            test = abs.(a .- anew)/(abs.(a) .+ abs.(anew))
            println("iteration: ",i," ",maximum(test))
            if maximum(test) < tol
                println("Solved in ",i," ","iterations")
                break
            end

    end
    return pol
end
h=Household()

pol=solve_bellman(h)

function inv_dist(h,apol,tol,maxit)
 @unpack agrid,Na, Ny,Pi,amax=h
    n=500
    na=500
    ny=Ny
    con_measure=ones(na,ny)
    start_ϵ = 1
    maxiter=100
    start_a = searchsortedfirst(agrid, 0.0)
    #initialize measures
    p_old = zeros(na,ny)
    p_new = zeros(na,ny)
    p_new[start_ϵ, start_a] = 1.0
    p_new[start_ϵ, start_a] = 1.0
    #agrid=copy(a_grid)
    fgrid=range(0.0,amax,length=n)
    #fpol=zeros(na,ny,n)
    #loc=similar(fpol)
    #p_new=ones(n,ny)./(n)
    #p_new=zeros(n,ny)
    iter=0
    diff=1.0
    while diff>tol && iter<maxit
        p_old=copy(p_new)/sum(sum(p_new))
        pnew=zeros(n,ny)
            for i=1:ny
                for j=1:n
                    itp=LinearInterpolation(agrid,apol[:,i];extrapolation_bc=Line())
                    #x=interpEGM(agrid,apol[:,i,t],fgrid[j],na)[1]
                    #fgrid[j,i,t]=x
                    x=itp(fgrid[j])
                    np=searchsortedlast(fgrid,x)
                    npn=min(np+1,n)
                    if np==0
                        np=1
                        w=1.0
                    elseif np==npn
                        w=1.0
                    else
                        w=(x-fgrid[np])/(fgrid[npn]-fgrid[np])
                    end

                    for ip=1:ny
                    #w=(x-fgrid[np])/(fgrid[np+1]-fgrid[np])
                    p_new[np,ip]= Pi[i,ip]*p_old[j,i]*(1.0-w) +p_new[np,ip]
                    p_new[npn,ip]=Pi[i,ip]*p_old[j,i]*w +p_new[npn,ip]
                    end
                end
            end
        diff=maximum(abs.(p_old-p_new))
        println("iteration: ",iter,diff)
        iter=iter+1


    end
    return p_old
end
p=inv_dist(h,pol',1e-8,100)
plot(p)
