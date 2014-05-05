export SVMLearner

type SVMLearner{T<:FP} <: AbstractLearner
    lambda::Float64
    p::Int
    coefs::Vector{T}
    gr::Vector{T}
    optimizer::AbstractSGD
    initialized::Bool
    function SVMLearner(optimizer::AbstractSGD, lambda::Float64)
        lambda >= 0.0 || error("lambda should be non-negative")
        obj = new()
        obj.optimizer = optimizer
        obj.lambda = lambda
        obj.initialized = false
        obj
    end
end

SVMLearner(optimizer::AbstractSGD, lambda::Float64) = SVMLearner{Float64}(optimizer, lambda)

function Base.show{T}(io::IO, obj::SVMLearner{T})
    print(io, "SVMLearner{T} lambda: $(obj.lambda)")
    print(io, "\nOptimizer: ")
    show(io, obj.optimizer)
    if obj.initialized
        print(io, "\nCoefficients: ")
        show(io, obj.coefs)
    end
end

function init!{T<:FP}(obj::SVMLearner{T}, p)
    obj.initialized && error("already initialized")
    obj.coefs = zeros(T, p)
    obj.gr = Array(T, p)
    obj.initialized = true
end

function grad!(obj::SVMLearner, x, y)
   error()
end

function update!{T <:FP}(obj::SVMLearner{T}, x::Matrix{T}, y::Vector{T})
    obj.initialized || init!(obj, size(x, 2))
    grad!(obj, x, y)
    update!(obj.optimizer, obj.coefs, obj.gr)
    obj
end
