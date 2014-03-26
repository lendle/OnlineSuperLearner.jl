
export AbstractSGD, SimpleSGD

abstract AbstractSGD
#should implement a update! method:
#update!{T<:FloatingPoint}(obj::AbstractSGD, weights::Matrix{T}, gr::Matrix{T})

type SimpleSGD <: AbstractSGD
    alpha1::Float64
    alpha2::Float64
    t::Int
    function SimpleSGD(alpha1::Float64, alpha2::Float64)
        alpha1 <= 0.0 || alpha2 <= 0.0 && error("alpha1 and alpha2 should be positive")
        new(alpha1, alpha2, 0)
    end
end

function update!{T<:FloatingPoint}(obj::SimpleSGD, weights::Matrix{T}, gr::Matrix{T})
    obj.t += 1
    stepsize = - obj.alpha1 * obj.alpha2/(obj.t + obj.alpha2)
    fma!(weights, convert(T, gr, stepsize))
    weights
end