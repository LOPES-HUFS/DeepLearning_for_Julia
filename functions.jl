function numerical_gradient(f, x, t, net)
    h = 1e-4
    vec = zeros(Float64, size(net)[1], size(net)[2])
    for i in 1:length(net)
        origin = net[i]
        net[i] += h
        fxh1 = f(x, t)
        net[i] -= (2 * h)
        fxh2 = f(x, t)
        vec[i] = (fxh1 - fxh2) / (2*h)
        net[i] = origin
    end
    return vec
end

function cross_entropy_error(y, t)
    delta = 1e-7
    return (sum(-sum.(t.* log.(y.+delta)))/size(t)[1])
end

function cross_entropy_error_single(y, t)
    delta = 1e-7
  return(-sum(t.* log.(y.+delta)))
end

function sigmoid(x)
    return 1/ (1+ exp(-x))
end

function softmax_single(a)
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end

function softmax(a)
    temp = map(softmax_single, eachrow(a))
    return(transpose(hcat(temp ...)))
end

function relu(x)
    if x > 0
        return x
    else
        return 0
    end
end
