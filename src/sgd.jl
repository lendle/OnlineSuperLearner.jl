
export AbstractSGD, SimpleSGD, AdaDelta, AdaGrad

abstract AbstractSGD
#should implement a update! method:
#update!(obj::AbstractSGD, weights::Vector{Float64}, gr::Vector{Float64})

type SimpleSGD <: AbstractSGD
    alpha1::Float64
    alpha2::Float64
    t::Int
    function SimpleSGD(alpha1::Float64, alpha2::Float64)
        alpha1 <= 0.0 || alpha2 <= 0.0 && error("alpha1 and alpha2 should be positive")
        new(alpha1, alpha2, 0)
    end
end

function update!(obj::SimpleSGD, weights::Vector{Float64}, gr::Vector{Float64})
    obj.t += 1

    stepsize = - obj.alpha1 / (1.0 + obj.alpha1 * obj.alpha2 * obj.t)
    fma!(weights, gr, stepsize)
    weights
end

type AdaDelta <: AbstractSGD
    rho::Float64
    eps::Float64
    sqgr::Vector{Float64}
    squp::Vector{Float64}
    up::Vector{Float64}
    initialized::Bool
    function AdaDelta(rho::Float64, eps::Float64)
        (rho <= 0.0 || eps <= 0.0) && error("rho and epsilon should be positive")
        obj = new()
        obj.rho = rho
        obj.eps = eps
        obj.initialized = false
        obj
    end
end

Base.show(io::IO, obj::AdaDelta) = print(io, "AdaDelta(rho=$(obj.rho), eps=$(obj.eps))")

function init!(obj::AdaDelta, weights)
    obj.initialized && error("already initialized")
    obj.sqgr = zeros(weights)
    obj.squp = zeros(weights)
    obj.up = similar(weights)
    obj.initialized = true
end

function update!(obj::AdaDelta, weights::Vector{Float64}, gr::Vector{Float64})
    obj.initialized || init!(obj, weights)
    @devec obj.sqgr[:] = obj.rho .* obj.sqgr + (1.0 - obj.rho) .* gr .* gr #line 4
    @devec obj.up[:] = - sqrt(obj.squp .+ obj.eps) ./ sqrt(obj.sqgr .+ obj.eps) .* gr #line 5
    @devec obj.squp[:] = obj.rho .* obj.squp + (1.0 - obj.rho) .* obj.up  .* obj.up #line 6
    add!(weights, obj.up) #line 7
end

type AdaGrad <: AbstractSGD
    eta::Float64
    sqgr::Vector{Float64}
    initialized::Bool
    function AdaGrad(eta::Float64)
        eta > 0.0 || error("eta should be positive")
        obj = new()
        obj.eta = eta
        obj.initialized = false
        obj
    end
end


Base.show(io::IO, obj::AdaGrad) = print(io, "AdaGrad(eta=$(obj.eta))")

function init!(obj::AdaGrad, weights)
    obj.initialized && error("already initialized")
    obj.sqgr = fill!(similar(weights), 1.0e-8)
    obj.initialized = true
end

function update!(obj::AdaGrad, weights::Vector{Float64}, gr::Vector{Float64})
    obj.initialized || init!(obj, weights)
    @devec obj.sqgr[:] += gr .* gr
    @devec weights[:] -= obj.eta ./ sqrt(obj.sqgr) .* gr
    weights
end
