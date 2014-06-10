# OnlineSuperLearner

[![Build Status](https://travis-ci.org/lendle/OnlineSuperLearner.jl.png)](https://travis-ci.org/lendle/OnlineSuperLearner.jl)

A young implementation of an online mini-batch version of the super learner algorithm in julia.

## Candidate learners

The super learner algorithm takes a library of candidate learners and combines them using a weighted combination chosen through cross validation. In `OnlineSuperLearner.jl`, candidate learners and the combined super learner are of type `Learner`.

A `Learner` implements an `update!{T <:FP}(obj::Learner{T}, x::Matrix{T}, y::Vector{T})` and a `predict!{T<:FP}(obj::Learner{T}, pr::Vector{T}, x::Matrix{T})` method.

The `predict` method is also available, which returns a vector of predictions instead of calculating them in place.

### Available candidate learners

* `GLMLearner(m::GLMModel, optimizer::AbstractSGD)` - GLMs without regularization.
* `GLMNetLearner(m::GLMModel, optimizer::AbstractSGD, lambda1 = 0.0, lambda2 = 0.0)` - GLMs with l_1 and l_2 regularization.
* `SVMLearner` - support vector machine, not fully implemented

The type of GLM is specified by `GLMModel`. Choices are:

* `LinearModel()` for least squares
* `LogisticModel` for logistic regression
* `QuantileModel(tau=0.5)` for `tau`-quantile regression.

## Optimization

All of the candidate learners require an optimizer of some sort.
Currently, optimization is done with stochastic gradient descent or some variant by the `AbstractSGD` type.

An `AbstractSGD` implements an `update!{T<:FP}(obj::AbstractSGD{T}, weights::Vector{T}, gr::Vector{T})` method.
This takes the current value of the weight(coefficient) vector and gradient and updates the weight vector in place.
The `AbstractSGD` instance stores tuning parameters and step information, and may have additional storage additional storage for if necessary.

### Available optimizers:

* `SimpleSGD(alpha1::Float64, alpha2::Float64)` - Step size is `alpha1/(1.0 + alpha1 * alpha2 * t)`.
* `AdaDelta(rho::Float64, eps::Float64)` - Implementation of Algorithm 1 [here](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf).
* `AdaGrad(eta::Float64)` Stepsize is for weight `j` `eta /[sqrt(sum of grad_j^2 up to t) + 1.0e-8]`. [Paper](http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf)
* `AveragedSGD(alpha1::Float64, alpha2::Float64, t0::Int)` - Described in [section 5.3](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) with step size `alpha1/(1.0 + alpha1 * alpha2 * t)^(3/4)`


## SuperLearner

A `SuperLearner{L<:Learner}(candidates::Vector{L}, combiner::GLMLearner)` object takes a vector of `candidates` and a `combiner` (typically a linear model for continuous outcomes or logistic for binary outcomes). `candidates` maybe be different types of learners or the same type with different optimizers or tuning parameters. The super learner is useful for tuning both learner tuning parameters and optimization tuning paramters.

## Notes

This is a work in progress. The `GLMLearner` and `GLMNetLearner` seem to be working pretty well, as are the implementaitons of `AbstractSGD`, but the `SuperLearner` type is not thoroughly tested.

### TODO
* Allow for features (`x`) to be sparse
* Finish the SVM implementation, perhaps add Pegasos implementation
* Automatic transformations of features
* More useful interfaces/DataFrames interface

