#jewel module OnlineSuperLearner

export SuperLearner

type SuperLearner <: Learner
    candidates::Vector{Learner}
    combiner::AbstractGLMLearner
    Z::Matrix{Float64}
    t::Int
    initialized::Bool
    function SuperLearner{L<:Learner}(candidates::Vector{L}, combiner::GLMLearner)
        obj = new()
        obj.candidates = candidates
        obj.combiner = combiner
        obj.t = 0
        obj.initialized = false
        obj
    end
end

function init!(obj::SuperLearner, n)
    obj.initialized && error("already initialized")
    obj.Z = Array(Float64, n, size(obj.candidates, 1))
    init!(obj.combiner, size(obj.candidates, 1))
    obj.combiner.coefs[:] = 1.0 / size(obj.candidates, 1)
    obj.initialized = true
end

function update_combination!(obj::SuperLearner, x::Matrix{Float64}, y::Vector{Float64})
    for (j, cand) in enumerate(obj.candidates)
            predict!(cand, view(obj.Z, :, j), x)
    end
    update!(obj.combiner, obj.Z, y)
    obj.combiner.coefs = projectsimplex(obj.combiner.coefs)
end

function update_candidates!(obj::SuperLearner, x::Matrix{Float64}, y::Vector{Float64})
    for cand in obj.candidates
        update!(cand, x, y)
    end
end

function update!(obj::SuperLearner, x::Matrix{Float64}, y::Vector{Float64})
    obj.initialized || init!(obj, size(x, 1))
    obj.t += 1
    if obj.t > 1
        update_combination!(obj, x, y)
    end
    update_candidates!(obj, x, y)
end

function predict!(obj::SuperLearner, pr::DenseVector{Float64}, x::Matrix{Float64})
    if size(x, 1) == size(obj.Z, 1)
        Z = obj.Z
    else
        Z = Array(Float64, size(x, 1), length(obj.candidates))
    end

    for (j, cand) in enumerate(obj.candidates)
        predict!(cand, view(Z, :, j), x)
    end
    predict!(obj.combiner, pr, Z)
end