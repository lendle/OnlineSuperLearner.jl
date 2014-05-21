export GLMModel, LogisticModel, LinearModel, QuantileModel, GLMLearner, update!,
        grad!, grad_scratch!, predict!, loss, predict, GLMNetLearner, linpred


##############
## GLMModel ##
##############################
## Used to specify GLM kind ##
##############################
abstract GLMModel
#Subtypes should define the loss, predict!, and grad_scratch! functions
# if the defaults are not appropriate.


function linpred!(pr::DenseVector{Float64},
                  coefs::Vector{Float64},
                  x::Matrix{Float64};
                  offset::Vector{Float64} = emptyvector(Float64))
        A_mul_B!(pr, x, coefs)
        if !isempty(offset)
            add!(pr, offset)
        end
        pr
end

#
# default predict! for GLMModel uses identity link
#
predict!(m::GLMModel,
        pr::DenseVector{Float64},
        coefs::Vector{Float64},
        x::Matrix{Float64};
        offset::Vector{Float64} = emptyvector(Float64)) = linpred!(pr, coefs, x; offset=offset)


# grad_scratch! calculates the gradient without allocating new memory
# for the residual vector.
# resid is scratch space and should have dims equal to size(y)
# Default is appropriate for GLMs with canonical link
function grad_scratch!(m::GLMModel,
                      gr::Vector{Float64},
                      coefs::Vector{Float64},
                      x::Matrix{Float64},
                      y::Vector{Float64},
                      resid::Vector{Float64};
                      offset::Vector{Float64} = emptyvector(Float64))
    predict!(m, resid, coefs, x; offset=offset)
    subtract!(resid, y) # resid = prediction - y
    alpha = 2.0/size(x, 1)
    BLAS.gemv!('T', alpha, x, resid, 0.0, gr) #grad \propto x'resid = -x'(y - pred)
    gr
end

grad!(m::GLMModel,
    gr::Vector{Float64},
    coefs::Vector{Float64},
    x::Matrix{Float64},
    y::Vector{Float64};
    offset::Vector{Float64} = emptyvector(Float64)) = grad_scratch!(m, gr, coefs, x, y, similar(y), offset=offset)

#########
## OLS ##
#########
type LinearModel <: GLMModel end

loss(m::LinearModel, pr::DenseVector{Float64}, y::Vector{Float64}) = meansqdiff(pr, y)


#########################
## Logistic regression ##
#########################
type LogisticModel <: GLMModel end

function predict!(m::LogisticModel,
                           pr::DenseVector{Float64},
                           coefs::Vector{Float64},
                           x::Matrix{Float64};
                           offset::Vector{Float64} = emptyvector(Float64))
    linpred!(pr, coefs, x; offset=offset)
    map1!(LogisticFun(), pr)
    pr
end


loss(m::LogisticModel, pr::DenseVector{Float64}, y::Vector{Float64}) = mean(NBLL(), pr, y)

#########################
## Quantile regression ##
#########################
type QuantileModel <: GLMModel
    tau :: Float64
end

QuantileModel() = QuantileModel(0.5)

function grad_scratch!(m::QuantileModel,
                               gr::Vector{Float64},
                               coefs::Vector{Float64},
                               x::Matrix{Float64},
                               y::Vector{Float64},
                               scratch::Vector{Float64};
                               offset::Vector{Float64} = emptyvector(Float64))
    predict!(m, scratch, coefs, x; offset=offset)
    subtract!(scratch, y)
    map1!(Qgrad(), scratch, m.tau)
    alpha = 2.0/size(x, 1)
    BLAS.gemv!('T', alpha, x, scratch, 0.0, gr)
    gr
end

loss(m::QuantileModel, pr::DenseVector{Float64}, y::Vector{Float64}) = meanabsdiff(pr, y)


##################
## End GLMModel ##
##################

abstract AbstractGLMLearner <: Learner


predict!(obj::AbstractGLMLearner, pr::DenseVector{Float64}, x::Matrix{Float64}; offset=emptyvector(Float64)) =
    predict!(obj.m, pr, obj.coefs, x, offset=offset)

predict(obj::AbstractGLMLearner, x::Matrix{Float64}; offset=emptyvector(Float64)) =
    predict!(obj, Array(Float64, size(x, 1)), x, offset=offset)

linpred!(obj::AbstractGLMLearner, pr::DenseVector{Float64}, x::Matrix{Float64}; offset=emptyvector(Float64)) =
    linpred!(pr, obj.coefs, x, offset=offset)

linpred(obj::AbstractGLMLearner, x::Matrix{Float64}; offset=emptyvector(Float64)) =
    linpred!(obj, Array(Float64, size(x, 1)), x, offset=offset)



################################
## GLM without regularization ##
################################
type GLMLearner <: AbstractGLMLearner
    m::GLMModel
    p::Int
    coefs::Vector{Float64}
    gr::Vector{Float64}
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


function Base.show(io::IO, obj::GLMLearner)
    print(io, "Model: ")
    show(io, obj.m)
    print(io, "\nOptimizer: ")
    show(io, obj.optimizer)
    if obj.initialized
        print(io, "\nCoefficients: ")
        show(io, obj.coefs)
    end
end

#Allocate obj.coefs and obj.gr on first call to update!
function init!(obj::GLMLearner, p)
    obj.initialized && error("already initialized")
    obj.p = p
    obj.coefs = zeros(p)
    obj.gr = Array(Float64, p)
    obj.initialized = true
    obj
end

function update!(obj::GLMLearner, x::Matrix{Float64}, y::Vector{Float64}; offset=emptyvector(Float64))
    obj.initialized || init!(obj, size(x, 2))
    grad!(obj.m, obj.gr, which_weights(obj.optimizer, obj.coefs), x, y, offset=offset)
    update!(obj.optimizer, obj.coefs, obj.gr)
    obj
end

#############################
## GLM with regularization ##
#############################

type GLMNetLearner <: AbstractGLMLearner
    m::GLMModel
    lambda1::Float64
    lambda2::Float64
    p::Int
    coefs::Vector{Float64}
    gr::Vector{Float64}
    gr2::Vector{Float64}
    mu::Vector{Float64}
    nu::Vector{Float64}
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

GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1 = 0.0, lambda2 = 0.0) = GLMNetLearner(m, optimizer, lambda1, lambda2)

function init!(obj::GLMNetLearner, p)
    obj.initialized && error("already initialized")
    obj.p = p
    obj.coefs = zeros(p)
    obj.gr = Array(Float64, p)
    if obj.lambda1 > 0.0
        obj.mu = zeros(p)
        obj.nu = zeros(p)
        obj.gr2 = Array(Float64, p)
        obj.optimizer2 = deepcopy(obj.optimizer)
    end
    obj.initialized = true
    obj
end

function update!(obj::GLMNetLearner, x::Matrix{Float64}, y::Vector{Float64})
    obj.initialized || init!(obj, size(x,2))

    #calculate the gradient like usual
    grad!(obj.m, obj.gr,  which_weights(obj.optimizer, obj.coefs), x, y)
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
