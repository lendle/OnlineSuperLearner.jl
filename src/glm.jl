export GLMModel, LogisticModel, LinearModel, QuantileModel, GLMLearner, update!,
        grad!, grad_scratch!, predict!, loss, predict, GLMNetLearner



abstract GLMModel
#Subtypes of GLMModel should define the loss, predict!, and grad_scratch! functions
# if hte defaults are not appropriate.


function linpred!{T <: FP}(m::GLMModel,
                           pr::Vector{T},
                           coefs::Vector{T},
                           x::Matrix{T};
                           offset::Vector{T} = emptyvector(T))
        A_mul_B!(pr, x, coefs)
        if !isempty(offset)
            add!(pr, offset)
        end
        pr
end

#
# this should be overridden if the GLMModel implementation does not use identity link
#
predict!{T <: FP}(m::GLMModel,
                  pr::Vector{T},
                  coefs::Vector{T},
                  x::Matrix{T};
                  offset::Vector{T} = emptyvector(T)) = linpred!(m, pr, coefs, x; offset=offset)


#grad_scratch! calculates the gradient without allocating new memory for the residual
# vector.
#resid is scratch space and  should have dims equal to size(y)
function grad_scratch!{T<: FP}(m::GLMModel,
                               gr::Vector{T},
                               coefs::Vector{T},
                               x::Matrix{T},
                               y::Vector{T},
                               resid::Vector{T};
                               offset::Vector{T} = emptyvector(T))
    predict!(m, resid, coefs, x; offset=offset)
    subtract!(resid, y) # resid = prediction - y
    alpha = 2.0/size(x, 1)
    BLAS.gemv!('T', convert(T, alpha), x, resid, zero(T), gr) #grad \propto x'resid = -x'(y - pred)
    gr
end

grad!{T<: FP}(m::GLMModel,
    gr::Vector{T},
    coefs::Vector{T},
    x::Matrix{T},
    y::Vector{T};
    offset::Vector{T} = emptyvector(T)) = grad_scratch!(m, gr, coefs, x, y, similar(y), offset=offset)

##
##OLS
##
type LinearModel <: GLMModel end

loss{T<: FP}(m::LinearModel, pr::Vector{T}, y::Vector{T}) = meansqdiff(pr, y)


##
##Logistic regression
##
type LogisticModel <: GLMModel end

function predict!{T <: FP}(m::LogisticModel,
                           pr::Vector{T},
                           coefs::Vector{T},
                           x::Matrix{T};
                           offset::Vector{T} = emptyvector(T))
    linpred!(m, pr, coefs, x; offset=offset)
    map1!(LogisticFun(), pr)
    pr
end


loss{T<: FP}(m::LogisticModel, pr::Vector{T}, y::Vector{T}) = mean(NBLL(), pr, y)

##
## Quantile regression
##
type QuantileModel <: GLMModel
    tau :: Float64
end

QuantileModel() = QuantileModel(0.5)

function grad_scratch!{T<: FP}(m::QuantileModel,
                               gr::Vector{T},
                               coefs::Vector{T},
                               x::Matrix{T},
                               y::Vector{T},
                               scratch::Vector{T};
                               offset::Vector{T} = emptyvector(T))
    predict!(m, scratch, coefs, x; offset=offset)
    subtract!(scratch, y)
    map1!(Qgrad(), scratch, m.tau)
    alpha = 2.0/size(x, 1)
    BLAS.gemv!('T', alpha, x, scratch, zero(T), gr)
    gr
end

loss{T<:FP}(m::QuantileModel, pr::Vector{T}, y::Vector{T}) = meanabsdiff(pr, y)


abstract AbstractGLMLearner <: AbstractLearner

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

type GLMLearner{T<:FP} <: AbstractGLMLearner
    m::GLMModel
    p::Int
    coefs::Vector{T}
    gr::Vector{T}
    optimizer::AbstractSGD
    initialized::Bool
    function GLMLearner(m::GLMModel, optimizer::AbstractSGD)
        obj = new()
        obj.m = m
        obj.optimizer = optimizer
        obj.initialized = false
        obj
    end
end

GLMLearner(m::GLMModel, optimizer::AbstractSGD) = GLMLearner{Float64}(m, optimizer)

#Allocate obj.coefs and obj.gr on first call to update!
function init!{T}(obj::GLMLearner{T}, p)
    obj.initialized && error("already initialized")
    obj.p = p
    obj.coefs = zeros(T, p)
    obj.gr = Array(T, p)
    obj.initialized = true
    obj
end

function update!{T <:FP}(obj::GLMLearner{T}, x::Matrix{T}, y::Vector{T})
    obj.initialized || init!(obj, size(x, 2))
    grad!(obj.m, obj.gr, obj.coefs, x, y)
    update!(obj.optimizer, obj.coefs, obj.gr)
    obj
end

predict!{T<:FP}(obj::AbstractGLMLearner, pr::Vector{T}, x::Matrix{T}; offset=emptyvector(T)) =
    predict!(obj.m, pr, obj.coefs, x, offset=offset)

predict{T<:FP}(obj::AbstractGLMLearner, x::Matrix{T}; offset=emptyvector(T)) =
    predict!(obj, Array(T, size(x,1)), x, offset=offset)

type GLMNetLearner{T<:FP} <: AbstractGLMLearner
    m::GLMModel
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
    function GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1::Float64, lambda2::Float64)
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


GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1 = 0.0, lambda2 = 0.0) = GLMNetLearner{Float64}(m, optimizer, lambda1, lambda2)


function init!{T}(obj::GLMNetLearner{T}, p)
    obj.initialized && error("already initialized")
    obj.p = p
    obj.coefs = zeros(T, p)
    obj.gr = Array(T, p)
    if obj.lambda1 > 0.0
        obj.mu = zeros(T, p)
        obj.nu = zeros(T, p)
        obj.gr2 = Array(T, p)
        obj.optimizer2 = deepcopy(obj.optimizer)
    end
    obj.initialized = true
    obj
end

function update!{T <:FP}(obj::GLMNetLearner{T}, x::Matrix{T}, y::Vector{T})
    obj.initialized || init!(obj, size(x,2))

    #calculate the gradient like usual
    grad!(obj.m, obj.gr, obj.coefs, x, y)
    if obj.lambda2 > 0.0
        #for l2 regularization, add l2 penalty
        fma!(obj.gr, obj.coefs, obj.lambda2) #gr = gr + lambda2 * coef
    end

    if obj.lambda1 > 0.0
        #for l1 regularization, calculate obj.gr - lambda1, stored in obj.gr2
        map!(Subtract(), obj.gr2, obj.lambda1, obj.gr)
        # then store obj.gr + lambda1 in obj.gr
        add!(obj.gr, obj.lambda1)
        # update the weights for both gradients in mu and nu
        update!(obj.optimizer, obj.mu, obj.gr)
        update!(obj.optimizer2, obj.nu, obj.gr2)
        #truncate truncate temp weights mu and nu at 0.0
        map1!(MaxFun(), obj.mu, 0.0)
        map1!(MaxFun(), obj.nu, 0.0)
        #calculat the final weights as mu - nu
        map!(Subtract(), obj.coefs, obj.mu, obj.nu)
    else
        #if no l1 regularization (so either only l2, or no reg at all)
        #just do standard update
        update!(obj.optimizer, obj.coefs, obj.gr)
    end
    obj
end
