module TestOnlineSuperLearner

using OnlineSuperLearner, Base.Test


my_tests = ["test_glm.jl",
        "test_sl.jl"]


println("Running tests:")


anyerrors = false

for my_test in my_tests
    try
        include(my_test)
        println("\t\033[1m\033[32mPASSED\033[0m: $(my_test)")
    catch
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(my_test)")
    end
end

if anyerrors
    throw("Tests failed")
end



end
