module TestOnlineSuperLearner

using OnlineSuperLearner, Base.Test

my_tests = ["test_glm.jl",
        "test_sl.jl"]

for my_test in my_tests
  include(my_test)
end

end
