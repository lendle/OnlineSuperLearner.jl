module OnlineSuperLearner

using NumericExtensions, NumericFuns, Devectorize, ArrayViews

import NumericExtensions.evaluate

import Common: predict, predict!, linpred, update!, loss

export Learner 

abstract Learner
#should implement an update! and a predict! method
# update!{T<:FloatingPoint}(obj::Learner{T}, x::Matrix{T}, y::Vector{T})
#and possibly additional kw args

predict(obj::Learner, x::Matrix; kwargs...) =
predict!(obj, Array(Float64, size(x,1)), x; kwargs...)


include("extra.jl")
include("sgd.jl")
include("glm.jl")
include("svm.jl")
include("sl.jl")
include("utils.jl")

end # module
