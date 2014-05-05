# In NumericFuns, don't ned
# type LogisticFun <: Functor{1} end
# evaluate(::LogisticFun, x) = one(x) / (one(x) + exp(-x))

#negative Bernoulli log likelihood
type NBLL <: Functor{2} end
evaluate(::NBLL, pr, y) = y == one(y)? -log(pr) :
                                 y == zero(y)? -log(one(y) - pr):
                                 -(y * log(pr) + (one(y) - y) * log(one(pr) - pr))

# Gradient helper for quantile regression
type Qgrad <: Functor{2} end
evaluate(::Qgrad, r, tau) = r == zero(r)? zero(r) :
                                 r < zero(r)? oftype(r, -tau):
                                 oftype(r, 1.0-tau)


emptyvector{T<:FP}(::Type{T}) = Array(T, 0)
