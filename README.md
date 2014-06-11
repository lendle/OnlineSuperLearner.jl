# OnlineSuperLearner

[![Build Status](https://img.shields.io/travis/lendle/OnlineSuperLearner.jl.svg?style=flat)](https://travis-ci.org/lendle/OnlineSuperLearner.jl)
[![Coverage Status](https://img.shields.io/coveralls/lendle/OnlineSuperLearner.jl.svg?style=flat)](https://coveralls.io/r/lendle/OnlineSuperLearner.jl)

A young implementation of an online mini-batch version of the super learner algorithm in julia.

## Candidate learners

The super learner algorithm takes a library of candidate learners and combines them using a weighted combination chosen through cross validation. In `OnlineSuperLearner.jl`, candidate learners and the combined super learner are of type `Learner`. `Learners` are provided by the [OnlineLearning.jl](https://github.com/lendle/OnlineLearning.jl) package.

## Optimization

All of the candidate learners require an optimizer of some sort.
Currently, optimization is done with stochastic gradient descent or some variant by the `AbstractSGD` type in the  [OnlineLearning.jl](https://github.com/lendle/OnlineLearning.jl) package.

## SuperLearner

A `SuperLearner{L<:Learner}(candidates::Vector{L}, combiner::GLMLearner)` object takes a vector of `candidates` and a `combiner` (typically a linear model for continuous outcomes or logistic for binary outcomes). `candidates` maybe be different types of learners or the same type with different optimizers or tuning parameters. The super learner is useful for tuning both learner tuning parameters and optimization tuning paramters.
