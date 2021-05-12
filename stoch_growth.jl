using QuantEcon, LinearAlgebra, Parameters,Optim,Interpolations
function parameters(β=0.95,α=0.33,δ=0.1,r=2,nk=101,nz=7,z_mean=1,ρ=0.67,σ=0.1)
  kstar=(α/(1/β - (1-δ)))^(1/(1-α))
  kgrid=range(0.1*kstar,5*kstar,length=nk)
  mc=rouwenhorst(nz,ρ,σ,z_mean)
  prob=mc.p
  z=mc.state_values
  K=repeat(kgrid,outer=[1,nz])
  v0=K.^α .- (1-δ).*K .+ K

  return (β=β,r=r,α=α,ρ=ρ,σ=σ,nk=nk,nz=nz,v0=v0,
          kgrid=kgrid,prob=prob,z=z,K=K,δ=δ,kstar=kstar)
end
sp=parameters()
vnext=zeros(sp.nk,sp.nz)
#v=copy(sp.v0)
#v_func=CubicSplineInterpolation(sp.kgrid,v[:,5])


function vnew(sp,v,x,a0,k0,j)
  @unpack r,α,β,δ,prob,nz=sp

  vp=zeros(1,nz)
    for i=1:nz
      v_func=LinearInterpolation(sp.kgrid,v[:,i];extrapolation_bc=Line())
      vp[1,i]=v_func(x)
    end
  # vp=v
   c=a0*k0^α+(1-δ)*k0-x
  if c <= 0
     val =-888888888888-800*abs(c)
   else
    val=c.^(1-r)/(1-r).+β.*((prob[j,:])'*vp')
   end
  return -val[1]
end


#vnext=zeros(sp.nk,sp.nz)
v=copy(sp.v0)
vnext=zeros(sp.nk,sp.nz)

tol=1e-3
dif=1
@time begin
while dif>tol
  for j in 1:sp.nz
    for i in 1:sp.nk
      a0=sp.z[j]
      k0=sp.kgrid[i,1]
      #object=(x->vnew(sp,v,x,a0,k0,j))
      #v_func=interp(sp.kgrid,v)
      #object= (kp -> (a0*k0^sp.α+(1-sp.δ)*k0-kp)^(1-sp.r)/(1-sp.r)+sp.β*(v_func.(kp)*sp.prob(j,:)'))
      results=optimize(x->vnew(sp,v,x,a0,k0,j),0.01,15)
      vnext[i,j] = results.minimum
    end
  end
  dif=norm(vnext.-v)
  display(dif)
  #v=vnext
  copy!(v,vnext)
end
end
