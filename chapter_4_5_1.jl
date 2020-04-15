# 줄리아가 타입 프로그래밍 언어라서 타입을 지정해주면 훨씬 빠르다.
# Float32로 바꿔서 돌려보기
# 로스 값의 의미, 절대적 의미? 상대적 의미?

"""
이 파일은 TwoLayerNet를 구성하고, MNIST 숫자 이미지를 가져온 다음
순전파로 TwoLayerNet를 학습시키는 스크립트입니다.
"""

include("functions.jl")
include("utils.jl")


"""
TwoLayerNet를 mutable struct로 만듭니다.
"""
mutable struct init_network
    W1
    b1
    W2
    b2
end

"""
앞에서 만든 struct를 초기화하는 함수를 작성합니다.
"""

# weight_init_std은 왜 곱하는가?
# 편향은 왜 0인가? (어떻게 바뀌는가?)
function making_network(input_size, hidden_size, output_size, weight_init_std =0.01)
    W1 = weight_init_std * randn(Float64, input_size, hidden_size)
    b1 = zeros(Float64, 1, hidden_size)
    W2 = weight_init_std * randn(Float64, hidden_size, output_size)
    b2 = zeros(Float64, 1, output_size)
    return(init_network(W1, b1, W2, b2))
end

"""
making_network() 이용하여 TwoLayerNet를 실제로 만듭니다. 
"""
TwoLayerNet = making_network(784, 50, 10)


# MNIST 자료를 가져와서 학습에 사용할 수 있게 전처리합니다.
# 아래에서 만든 x, t으로 학습을 하게 됩니다.


using MLDatasets

train_x, train_y = MNIST.traindata()
train_x = reshape(train_x, 784, 60000)
t = making_one_hot_labels(train_y)
train_x = Array{Float64}(train_x)
x = transpose(train_x)

"""
순전파로 학습하거나 학습된 네트웍을 평가하기 위한 함수입니다. 
"""
function predict(x)
    a1 = (x * TwoLayerNet.W1) .+ TwoLayerNet.b1
    z1 = sigmoid.(a1)
    a2 = (z1 * TwoLayerNet.W2) .+ TwoLayerNet.b2
    return (softmax(a2))
end

"""
앞에서 만든 `predict()`를 `numerical_gradient()`에서 처리하기 위해서
함수를 만듭니다.
"""
function f(x, t)
    y = predict(x)
    return cross_entropy_error(y, t)
end

function TwoLayerNet_numerical_gradient(f, x, t)
    W1 = numerical_gradient(f, x, t,TwoLayerNet.W1)
    W2 = numerical_gradient(f, x, t,TwoLayerNet.W2)
    b1 = numerical_gradient(f, x, t,TwoLayerNet.b1)
    b2 = numerical_gradient(f, x, t,TwoLayerNet.b2)
    return(init_network(W1, b1, W2, b2))
end

"""
손실함수
"""
function loss(x, t)
    y = predict(x)
    return cross_entropy_error(y, t)
end

# 지금까지 만든 것들은 이용해서 순전파를 이용해서 100개의 자료로
# 학습시켜봅니다.

@time begin
    TwoLayerNet_grads = TwoLayerNet_numerical_gradient(f, x[1:100, :], t[1:100])
end

# 학습한 네트웍을 평가하기 위한 자료와 함수를 만듭니다.

test_x,  test_y  = MNIST.testdata()
# x 자료를 다루기 위해서 변경합니다.
x_test = reshape(test_x, 784, 10000)
x_test = transpose(x_test)
x_test = Array{Float64}(x_test)

"""
학습한 네트웍을 평가하기 위한 함수
"""
function evaluate(x, t)
    temp =sum((argmax.(eachrow(predict(x))).-1) .== t)/size(x)[1]
    return (temp *100)
end

# 지금까지 만든 것들은 이용해서 순전파를 이용해서 100개씩 자료를 입력해서
# 학습시켜봅니다.

train_size = size(x)[1]
batch_size = 100
learning_rate = 0.1

# 학습시킬 때 학습률와 정확도를 저장하는 함수를 만듭니다.
train_loss_list = Float64[]
accuracy = Float64[]

# 배치로 100씩, 100번을 돌려 봅니다.
iters_num = 100

@time begin
    for i in 1:iters_num
        batch_mask = rand(1:train_size, 100)
        x_batch = x[batch_mask, :]
        t_batch = t[batch_mask, :]
        grad = TwoLayerNet_numerical_gradient(f, x_batch, t_batch)
    
        TwoLayerNet.W1 -= (learning_rate * grad.W1)
        TwoLayerNet.W2 -= (learning_rate * grad.W2)
        TwoLayerNet.b1 -= (learning_rate * grad.b1)
        TwoLayerNet.b2 -= (learning_rate * grad.b2)
    
        temp_loss = loss(x_batch, t_batch)
        print("NO.$i: ")
        println(temp_loss)
        append!(train_loss_list, temp_loss)
        append!(accuracy, evaluate(x_test, test_y))
    end
end
