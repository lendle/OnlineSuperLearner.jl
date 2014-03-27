type LogisticFun <: Functor{1} end
evaluate(::LogisticFun, x) = one(x) / (one(x) + exp(-x))

type NBLL <: Functor{2} end
evaluate(::NBLL, pr, y) = y == one(y)? -log(pr) :
                                 y == zero(y)? -log(one(y) - pr):
                                 -(y * log(pr) + (one(y) - y) * log(one(pr) - pr))

