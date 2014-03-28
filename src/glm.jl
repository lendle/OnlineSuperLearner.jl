export GLModel, LogisticModel, LinearModel, QuantileModel, GLMLearner, update!,
        grad!, grad_scratch!, predict!, loss, predict, GLMNetLearner

abstract AbstractGLMLearner

abstract GLModel
#Subtypes of GLModel should define the loss and predict! functions

emptyvec{T<:FP}(::Type{T}) = Array(T, 0)

function linpred!{T <: FP}(m::GLModel,
                           pr::Vector{T},
                           coefs::Vector{T},
                           x::Matrix{T};
                           offset::Vector{T} = emptyvec(T))
        BLAS.gemv!('N', one(T), x, coefs, zero(T), pr)
        if !isempty(offset)
            add!(pr, offset)
        end
        pr
end

#
# this should be overridden if the subtype does not use identity link
#
predict!{T <: FP}(m::GLModel,
                  pr::Vector{T},
                  coefs::Vector{T},
                  x::Matrix{T};
                  offset::Vector{T} = emptyvec(T)) = linpred!(m, pr, coefs, x; offset=offset)


#grad_scratch! calculates the gradient without allocating new memory for the residual
# vector.
#resid is scratch space and  should have dims equal to size(y)
function grad_scratch!{T<: FP}(m::GLModel,
                               gr::Vector{T},
                               coefs::Vector{T},
                               x::Matrix{T},
                               y::Vector{T},
                               resid::Vector{T};
                               offset::Vector{T} = emptyvec(T))
    predict!(m, resid, coefs, x; offset=offset)
    subtract!(resid, y) # resid = prediction - y
    alpha = 2.0/size(x, 1)
    BLAS.gemv!('T', convert(T, alpha), x, resid, zero(T), gr) #grad \propto x'resid = -x'(y - pred)
    gr
end

grad!{T<: FP}(m::GLModel,
    gr::Vector{T},
    coefs::Vector{T},
    x::Matrix{T},
    y::Vector{T};
    offset::Vector{T} = emptyvec(T)) = grad_scratch!(m, gr, coefs, x, y, similar(y), offset=offset)

##
##OLS
##
type LinearModel <: GLModel end

loss{T<: FP}(m::LinearModel, pr::Vector{T}, y::Vector{T}) = meansqdiff(pr, y)


##
##Logistic regression
##
type LogisticModel <: GLModel end

function predict!{T <: FP}(m::LogisticModel,
                           pr::Vector{T},
                           coefs::Vector{T},
                           x::Matrix{T};
                           offset::Vector{T} = emptyvec(T))
    linpred!(m, pr, coefs, x; offset=offset)
    map1!(LogisticFun(), pr)
    pr
end


loss{T<: FP}(m::LogisticModel, pr::Vector{T}, y::Vector{T}) = mean(NBLL(), pr, y)

##
## Quantile regression
##
type QuantileModel <: GLModel
    tau :: Float64
end

QuantileModel() = QuantileModel(0.5)

function grad_scratch!{T<: FP}(m::QuantileModel,
                               gr::Vector{T},
                               coefs::Vector{T},
                               x::Matrix{T},
                               y::Vector{T},
                               scratch::Vector{T};
                               offset::Vector{T} = emptyvec(T))
    predict!(m, scratch, coefs, x; offset=offset)
    subtract!(scratch, y)
    map1!(Qgrad(), scratch, m.tau)
    alpha = 2.0/size(x, 1)
    BLAS.gemv!('T', alpha, x, scratch, zero(T), gr)
    gr
end

loss{T<:FP}(m::QuantileModel, pr::Vector{T}, y::Vector{T}) = meanabsdiff(pr, y)


type GLMLearner{T<:FP} <: AbstractGLMLearner
    m::GLModel
    p::Int
    coefs::Vector{T}
    gr::Vector{T}
    optimizer::AbstractSGD
    initialized::Bool
    function GLMLearner(m::GLModel, optimizer::AbstractSGD)
        obj = new()
        obj.m = m
        obj.optimizer = optimizer
        obj.initialized = false
        obj
    end
end

GLMLearner(m::GLModel, optimizer::AbstractSGD) = GLMLearner{Float64}(m, optimizer)

function Base.show(io::IO, obj::AbstractGLMLearner)
    print(io, "Model: ")
    show(io, obj.m)
    print(io, "\nOptimizer: ")
    show(io, obj.optimizer)
    if obj.initialized
        print(io, "\nCoefficients: ")
        show(io, obj.coefs)
    end
end

function init!{T}(obj::GLMLearner{T}, x)
    obj.initialized && error("already initialized")
    obj.p = size(x, 2)
    obj.coefs = zeros(T, obj.p)
    obj.gr = Array(T, obj.p)
    obj.initialized = true
    obj
end

function update!{T <:FP}(obj::GLMLearner{T}, x::Matrix{T}, y::Vector{T})
    obj.initialized || init!(obj, x)
    grad!(obj.m, obj.gr, obj.coefs, x, y)
    update!(obj.optimizer, obj.coefs, obj.gr)
    obj
end

predict!{T<:FP}(obj::AbstractGLMLearner, pr::Vector{T}, x::Matrix{T}; offset=emptyvec(T)) =
    predict!(obj.m, pr, obj.coefs, x, offset=offset)

predict{T<:FP}(obj::AbstractGLMLearner, x::Matrix{T}; offset=emptyvec(T)) =
    predict!(obj, Array(T, size(x,1)), x, offset=offset)

type GLMNetLearner{T<:FP} <: AbstractGLMLearner
    m::GLModel
    lambda1::Float64
    lambda2::Float64
    p::Int
    coefs::Vector{T}
    gr::Vector{T}
    gr2::Vector{T}
    mu::Vector{T}
    nu::Vector{T}
    optimizer::AbstractSGD
    optimizer2::AbstractSGD
    initialized::Bool
    function GLMNetLearner(m::GLModel, optimizer::AbstractSGD, lambda1::Float64, lambda2::Float64)
        lambda1 < 0.0 || lambda2 < 0.0 && error("lambda1 and lambda2 should be non-negative")
        obj = new()
        obj.m = m
        obj.lambda1 = lambda1
        obj.lambda2 = lambda2
        obj.optimizer = optimizer
        obj.initialized = false
        obj
    end
end


GLMNetLearner(m::GLModel, optimizer::AbstractSGD, lambda1 = 0.0, lambda2 = 0.0) = GLMNetLearner{Float64}(m, optimizer, lambda1, lambda2)


function init!{T}(obj::GLMNetLearner{T}, x)
    obj.initialized && error("already initialized")
    obj.p = size(x, 2)
    obj.coefs = zeros(T, obj.p)
    obj.gr = Array(T, obj.p)
    if obj.lambda1 > 0.0
        obj.mu = zeros(T, obj.p)
        obj.nu = zeros(T, obj.p)
        obj.gr2 = Array(T, obj.p)
        obj.optimizer2 = deepcopy(obj.optimizer)
    end
    obj.initialized = true
    obj
end

function update!{T <:FP}(obj::GLMNetLearner{T}, x::Matrix{T}, y::Vector{T})
    obj.initialized || init!(obj, x)
    grad!(obj.m, obj.gr, obj.coefs, x, y)
    if obj.lambda2 > 0.0
        fma!(obj.gr, obj.coefs, obj.lambda2) #gr = gr + lambda2 * coef
    end

    if obj.lambda1 > 0.0
        map!(Subtract(), obj.gr2, obj.lambda1, obj.gr)
        add!(obj.gr, obj.lambda1)
        update!(obj.optimizer, obj.mu, obj.gr)
        update!(obj.optimizer2, obj.nu, obj.gr2)
        map1!(MaxFun(), obj.mu, 0.0)
        map1!(MaxFun(), obj.nu, 0.0)
        map!(Subtract(), obj.coefs, obj.mu, obj.nu)
    else
        update!(obj.optimizer, obj.coefs, obj.gr)
    end
    obj
end
