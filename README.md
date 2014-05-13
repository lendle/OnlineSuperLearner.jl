# OnlineSuperLearner

[![Build Status](https://travis-ci.org/lendle/OnlineSuperLearner.jl.png)](https://travis-ci.org/lendle/OnlineSuperLearner.jl)

# Learners

The super learner algorithm takes a set of candidate learners and combines them using a weighted combination chosen through cross validation. In `SuperLearner.jl`, learners are of type `Learner`.

An `Learner` implements an `update!{T <:FP}(obj::Learner{T}, x::Matrix{T}, y::Vector{T})` and a `predict!{T<:FP}(obj::Learner{T}, pr::Vector{T}, x::Matrix{T})` method. There is also `predict`, which returns predictions instead of calculating them in place.


## `GLMLearner` - GLMs without regularization

Initialized with `GLMLearner(m::GLMModel, optimizer::AbstractSGD)`

## `GLMNetLearner` - GLMs with l_1 and l_2 regularization

Initialized with `GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1 = 0.0, lambda2 = 0.0)` where `lambda1` and `lambda2` are the l_1 and l_2 regularization parameters.

If `lambda1` is positive, a copy of the optimizer is made to be used on the two sets of weights.

## `SVMLearner` - support vector machines, not fully implemented


# Optimization

All of the learners require an optimizer of some sort.
Currently, optimization is done with stochastic gradient descent or some variant by the `AbstractSGD` type.

An `AbstractSGD` implements an `update!{T<:FP}(obj::AbstractSGD{T}, weights::Vector{T}, gr::Vector{T})` method.
This takes the current value of the weight(coefficient) vector and gradient and updates the weight vector in place.
The `AbstractSGD` instance stores tuning parameters and step information, as well as additional storage for if necessary.

Currently available:

* `SimpleSGD` - Requires two parameters `alpha1` and `alpha2`, step size is `alpha1/(1.0 + alpha1 * alpha2 * t)`.
* `AdaDelta` -- Parameters `rho` and `epsilon`. Implementation of Algorithm 1 [here](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf).
* `AdaGrad` -- Parameter `eta`. Stepsize is for weight `j` `eta /[sqrt(sum of grad_j^2 up to t) + 1.0e-8]`.



# GLMModel

Used to specify GLMs. `LinearModel()` for least squares, `LogisticModel` for logistic regression, and `QuantileModel(tau=0.5)` for `tau`-quantile regression.
Defaults to median regression.

