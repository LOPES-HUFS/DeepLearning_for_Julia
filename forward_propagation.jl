1. MNIST 데이터 다운 및 전처리

using MLDatasets
train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()


test_x = reshape(test_x,784, 10000)
test_x = Array{Float64}(test_x)
test_x=test_x'

train_x = reshape(train_x,784, 60000)
train_x = Array{Float64}(train_x)
train_x=train_x'

# t 자료를 원핫 레이블로 변경
t = making_one_hot_labels(train_y)
typeof(t), size(t)

2. 네트워크 및 초기 매개 변수 설정

"""
TwoLayerNet를 mutable struct로 만듭니다.
"""
mutable struct init_network
    W1
    b1
    W2
    b2
end

function making_network(input_size, hidden_size, output_size, weight_init_std =0.01)
    W1 = weight_init_std * randn(Float64, input_size, hidden_size)
    b1 = zeros(Float32, 1, hidden_size)
    W2 = weight_init_std * randn(Float64, hidden_size, output_size)
    b2 = zeros(Float32, 1, output_size)
    return(init_network(W1, b1, W2, b2))
end

TwoLayerNet = making_network(784, 50, 10)

3. 순전파에 필요한 함수 정의

function predict(x)
    a1 = (x * TwoLayerNet.W1) .+ TwoLayerNet.b1
    z1 = sigmoid.(a1)
    a2 = (z1 * TwoLayerNet.W2) .+ TwoLayerNet.b2
    return softmax(a2)
end

function loss(x, t)
    y = predict(x)
    return cross_entropy_error(y, t)
end

function cross_entropy_error(y,t)
    delta = 1e-7
    batch_size = length(y[:,1])
    return (-sum(log.(y.+delta).*t) / batch_size)
end

function sigmoid(x)
    return 1/(1+exp(-x))
end

function softmax_single(a)
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end

function softmax(a)
    temp = map(softmax_single, eachrow(a))
    return(transpose(hcat(temp ...)))
end


# f는 손실함수, x는 입력값, t는 정답, w는 대상
function numerical_gradient(f, x, t, w)
    h=10^-4
    vec=zeros(Float64,size(w))
    
for i in (1:length(w))
        origin=w[i]
        w[i]+=h
        fx1=f(x,t)
        w[i]-=2*h
        fx2=f(x,t)
        vec[i]=(fx1-fx2)/2h
        w[i]=origin
    end
    return  vec
end

function TwoLayerNet_numerical_gradient(f, x, t)
    W1 = numerical_gradient(f, x, t,TwoLayerNet.W1)
    W2 = numerical_gradient(f, x, t,TwoLayerNet.W2)
    b1 = numerical_gradient(f, x, t,TwoLayerNet.b1)
    b2 = numerical_gradient(f, x, t,TwoLayerNet.b2)
    return(init_network(W1, b1, W2, b2))
end

function evaluate(x, t)
    temp = (sum((argmax.(eachrow(predict(test_x))).-1) .== test_y)/size(test_x)[1])
    return (temp * 100)
end

4. 순전파에 필요한 변수 정의

train_size = size(train_x)[1]
batch_size = 100
learning_rate = 0.1
train_loss_list = Float64[]
accuracy = Float64[]

5. 순전파 알고리즘 코드

@time begin
    for i in 1:600
        batch_mask = rand(1:train_size, 100)
        x_batch = train_x[batch_mask, :]
        t_batch = t[batch_mask, :]
        grad = TwoLayerNet_numerical_gradient(loss, x_batch, t_batch)
    
        TwoLayerNet.W1 -= (learning_rate * grad.W1)
        TwoLayerNet.W2 -= (learning_rate * grad.W2)
        TwoLayerNet.b1 -= (learning_rate * grad.b1)
        TwoLayerNet.b2 -= (learning_rate * grad.b2)
    
        temp_loss = loss(x_batch, t_batch)
        print("NO.$i: ")
        println(temp_loss)
        append!(train_loss_list, temp_loss)
        append!(accuracy, evaluate(test_x, test_y))
    end
end

6. 손실 함수, 정확도 그래프 그리기

using Plots

# 손실 함수
x = range(1,length(train_loss_list),step=1)
y = train_loss_list
plot(x,y)

# 정확도
x = range(1,length(accuracy),step=1)
y = accuracy
plot(x,y)
