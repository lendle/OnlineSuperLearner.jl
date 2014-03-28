
export AbstractSGD, SimpleSGD, AdaDelta, AdaGrad

abstract AbstractSGD
#should implement a update! method:
#update!{T<:FP}(obj::AbstractSGD, weights::Vector{T}, gr::Vector{T})

type SimpleSGD <: AbstractSGD
    alpha1::Float64
    alpha2::Float64
    t::Int
    function SimpleSGD(alpha1::Float64, alpha2::Float64)
        alpha1 <= 0.0 || alpha2 <= 0.0 && error("alpha1 and alpha2 should be positive")
        new(alpha1, alpha2, 0)
    end
end

function update!{T<:FP}(obj::SimpleSGD, weights::Vector{T}, gr::Vector{T})
    obj.t += 1
    stepsize = - obj.alpha1 * obj.alpha2/(obj.t + obj.alpha2)
    fma!(weights, gr, convert(T, stepsize))
    weights
end

type AdaDelta{T<:FP} <: AbstractSGD
    rho::Float64
    eps::Float64
    sqgr::Vector{T}
    squp::Vector{T}
    up::Vector{T}
    initialized::Bool
    function AdaDelta(rho::Float64, eps::Float64)
        rho <= 0.0 || eps <= 0.0 && error("rho and epsilon should be positive")
        obj = new()
        obj.rho = rho
        obj.eps = eps
        obj.initialized = false
        obj
    end
end

AdaDelta(rho::Float64, eps::Float64) = AdaDelta{Float64}(rho, eps)

Base.show{T}(io::IO, obj::AdaDelta{T}) = print(io, "AdaDelta{$T}(rho=$(obj.rho), eps=$(obj.eps))")

function init!(obj::AdaDelta, weights)
    obj.initialized && error("already initialized")
    obj.sqgr = zeros(weights)
    obj.squp = zeros(weights)
    obj.up = similar(weights)
    obj.initialized = true
end

function update!{T<:FP}(obj::AdaDelta, weights::Vector{T}, gr::Vector{T})
    obj.initialized || init!(obj, weights)
    @devec obj.sqgr[:] = obj.rho .* obj.sqgr + (1.0 - obj.rho) .* gr .* gr #line 4
    @devec obj.up[:] = - sqrt(obj.squp .+ obj.eps) ./ sqrt(obj.sqgr .+ obj.eps) .* gr #line 5
    @devec obj.squp[:] = obj.rho .* obj.squp + (1.0 - obj.rho) .* obj.up  .* obj.up #line 6
    add!(weights, obj.up) #line 7
end

type AdaGrad{T<:FP} <: AbstractSGD
    eta::Float64
    sqgr::Vector{T}
    initialized::Bool
    function AdaGrad(eta::Float64)
        eta > 0.0 || error("eta should be positive")
        obj = new()
        obj.eta = eta
        obj.initialized = false
        obj
    end
end

AdaGrad(eta::Float64) = AdaGrad{Float64}(eta)

function init!(obj::AdaGrad, weights)
    obj.initialized && error("already initialized")
    obj.sqgr = zeros(weights)
    obj.initialized = true
end

function update!{T<:FP}(obj::AdaGrad, weights::Vector{T}, gr::Vector{T})
    obj.initialized || init!(obj, weights)
    @devec obj.sqgr[:] += gr .* gr
    @devec weights[:] -= obj.eta ./ sqrt(obj.sqgr) .* gr
    weights
end
