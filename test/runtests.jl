using OnlineSuperLearner, Base.Test


mylearner=  GLMLearner(LogisticModel(), AdaDelta(1.0, 1.0))

x = rand(10, 10)
y = round(rand(10))

update!(mylearner, x, y)

predict(mylearner, x)

mylearner32 =  GLMLearner{Float32}(LogisticModel(), AdaDelta{Float32}(1.0, 1.0))

x32 = rand(Float32, 10,10)
y32 = round(rand(Float32, 10))
update!(mylearner32, x32, y32)
predict(mylearner32, x32)
