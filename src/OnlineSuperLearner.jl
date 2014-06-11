module OnlineSuperLearner

using NumericExtensions, NumericFuns, Devectorize, ArrayViews, Reexport

import NumericExtensions.evaluate

importall OnlineLearning
import OnlineLearning: DSMat, init!
@reexport using OnlineLearning

include("extra.jl")
include("sl.jl")

end # module
