# OnlineSuperLearner

[![Build Status](https://travis-ci.org/lendle/OnlineSuperLearner.jl.png)](https://travis-ci.org/lendle/OnlineSuperLearner.jl)

# Types

## AbstractSGD

An `AbstractSGD` type implements an `update!{T<:FP}(obj::SimpleSGD, weights::Vector{T}, gr::Vector{T})` method.
This takes the current value of the weight(coefficient) vector and gradient and updates the weight vector in place.
The `AbstractSGD` instance stores tuning parameters and step information, as well as additional storage for if necessary.

* `SimpleSGD` --- Requires two parameters `alpha1` and `alpha2`, step size is `alpha1/(1.0 + alpha1 * alpha2 * t)`.
* `AdaDelta` -- Parameters `rho` and `epsilon`. Implementation of Algorithm 1 [here](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf).
* `AdaGrad` -- Parameter `eta`. Stepsize is for weight `j` `eta /[sqrt(sum of grad_j^2 over steps) + 1.0e-8]`.



## AbstractLearner

An `AbstractLearner` should implement an `update!` method that takes a matrix `x` and outcome vector `y`, and a predict method that takes a matrix `x`.

### `GLMLearner` --- GLMs without regularization

Initialized with `GLMLearner(m::GLMModel, optimizer::AbstractSGD)`

### `GLMNetLearner` --- GLMs with l_1 and l_2 regularization

Initialized with `GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1 = 0.0, lambda2 = 0.0)` where `lambda1` and `lambda2` are the l_1 and l_2 regularization parameters.

If `lambda1` is positive, a copy of the optimizer is made to be used on the two sets of weights.

* `SVMLearner` --- support vector machines, not fully implemented


## GLMModel

Used to specify GLMs. `LinearModel()` for least squares, `LogisticModel` for logistic regression, and `QuantileModel(tau=0.5)` for `tau`-quantil regression.
Defaults to median regression.

