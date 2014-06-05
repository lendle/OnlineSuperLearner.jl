export SuperLearner, calc_losses

type SuperLearner <: Learner
    candidates::Vector{Learner}
    combiner::GLMLearner
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

function Base.show(io::IO, sl::SuperLearner)
  Base.print(io, "Candidates: ")
  for cand in sl.candidates
      print(io, "\n\t")
      show(io, cand)
  end
  Base.print(io, "\nCombiner:\n\t")
  Base.show(io, sl.combiner)
  if sl.combiner.initialized
    Base.println(io, "\nWeights: ")
    Base.println(io, sl.combiner.coefs)
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
        #candidates must be updated at least once before updating combination
        update_combination!(obj, x, y)
    end
    update_candidates!(obj, x, y)
end

#if obj.Z is the right size, return it. Otherwise allocate a new matrix and return that.
get_Z(obj::SuperLearner, x::Matrix{Float64}) = size(x, 1) == size(obj.Z, 1)? obj.Z: Array(Float64, size(x, 1), length(obj.candidates))

function predict!(obj::SuperLearner, pr::DenseVector{Float64}, x::Matrix{Float64}, Z::Matrix{Float64}=get_Z(obj, x))
    for (j, cand) in enumerate(obj.candidates)
        predict!(cand, view(Z, :, j), x)
    end
    predict!(obj.combiner, pr, Z)
end

function calc_losses(obj::SuperLearner, x::Matrix{Float64}, y::Vector{Float64})
    pr = Array(Float64, size(x, 1))
    losses = Array(Float64, length(obj.candidates) + 1)
    Z = get_Z(obj, x)
    predict!(obj, pr, x, Z)
    for (j, cand) in enumerate(obj.candidates)
        losses[j] = loss(obj.combiner.m, view(Z, :, j), y)
    end
    losses[end] = loss(obj.combiner.m, pr, y)
    losses
end
