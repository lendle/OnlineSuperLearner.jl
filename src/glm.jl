export GLModel, LogisticModel, LinearModel, QuantileModel, GLMLearner, update!,
        grad!, grad_scratch!, predict!, loss, predict


abstract GLModel
#Subtypes of GLModel should define the loss and predict! functions

emptyvec{T<:FP}(::Type{T}) = Array(T, 0)

function linpred!{T <: FP}(m::GLModel,
                           pr::Vector{T},
                           coefs::Vector{T},
                           x::Matrix{T};
                           offset::Vector{T} = emptyvec(T))
        BLAS.gemv!('N', 1.0, x, coefs, 0.0, pr)
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
#scratch should have dims equal to size(y)
function grad_scratch!{T<: FP}(m::GLModel,
                               gr::Vector{T},
                               coefs::Vector{T},
                               x::Matrix{T},
                               y::Vector{T},
                               scratch::Vector{T};
                               offset::Vector{T} = emptyvec(T))
    predict!(m, scratch, coefs, x; offset=offset)
    subtract!(scratch, y)
    alpha = 2.0/size(x, 1)
    BLAS.gemv!('T', alpha, x, scratch, 0.0, gr)
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
    BLAS.gemv!('T', alpha, x, scratch, 0.0, gr)
    gr
end

loss{T<:FP}(m::QuantileModel, pr::Vector{T}, y::Vector{T}) = meanabsdiff(pr, y)


type GLMLearner{T<:FP}
    m::GLModel
    p::Int
    coefs::Vector{T}
    gr::Vector{T}
    optimizer::AbstractSGD
    initialized
    function GLMLearner{T <:FP}(m::GLModel, optimizer::AbstractSGD, ::Type{T})
        obj = new()
        obj.m = m
        obj.optimizer = optimizer
        obj.initialized = false
        obj
    end
end

GLMLearner{T<:FP}(m::GLModel, optimizer::AbstractSGD, ::Type{T}) = GLMLearner{T}(m, optimizer, T)

GLMLearner(m::GLModel, optimizer::AbstractSGD) = GLMLearner(m, optimizer, Float64)


function init!{T}(obj::GLMLearner{T}, x)
    obj.initialized && error("obj already initialized")
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

predict!{T<:FP}(obj::GLMLearner{T}, pr::Vector{T}, x::Matrix{T}; offset=emptyvec(T)) =
    predict!(obj.m, pr, obj.coefs, x, offset=offset)

predict{T<:FP}(obj::GLMLearner, x::Matrix{T}; offset=emptyvec(T)) =
    predict!(obj, Array(T, size(x,1)), x, offset=offset)
