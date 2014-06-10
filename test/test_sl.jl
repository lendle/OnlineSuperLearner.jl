cands  = [GLMLearner(LogisticModel(), SimpleSGD(1.0, 1.0)),
          GLMLearner(LogisticModel(), SimpleSGD(1.0, 0.1)),
          GLMNetLearner(LogisticModel(), SimpleSGD(1.0, 1.0), 1.0, 1.0)]

combiner = GLMLearner(LogisticModel(), SimpleSGD(1.0, 1.0))

osl = SuperLearner(cands, combiner)

update!(osl, x, y)

update!(osl, x, y)

predict(osl, x)
