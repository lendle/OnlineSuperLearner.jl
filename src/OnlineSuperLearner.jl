module OnlineSuperLearner

using NumericExtensions, NumericFuns, Devectorize

import NumericExtensions.evaluate

typealias FP FloatingPoint

abstract AbstractLearner
#should implement an update and a predict method
# update!{T<:FloatingPoint}(obj::AbstractLearner, x::Matrix{T}, y::Vector{T})
#and possibly additional kw args

include("utils.jl")
include("sgd.jl")
include("glm.jl")
include("svm.jl")


end # module
