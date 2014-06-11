function projectsimplex{T <: Real}(v::Array{T, 1}, z::T=one(T))
  n=length(v)
  µ = sort(v, rev=true)
  #finding ρ could be improved to avoid so much temp memory allocation
  ρ = maximum((1:n)[µ - (cumsum(µ) .- z) ./ (1:n) .>0])
  θ = (sum(µ[1:ρ]) - z)/ρ
  max(v .- θ, 0)
end
