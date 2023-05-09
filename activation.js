function sigmoid(x){
    return 1 / (1 + Math.exp(-x));
}

function sigmoidPrime(x){
    return sigmoid(x) * (1 - sigmoid(x));
}

function tanh(x){
    return Math.tanh(x);
}

function tanhPrime(x){
    return 1 - Math.pow(Math.tanh(x), 2);
}

function linear(x){
    return x;
}

function linearPrime(x){
    return 1;
}

function relu(x){
    return x < 0 ? 0 : x;
}

function reluPrime(x){
    return x < 0 ? 0 : 1;
}

function leakyRelu(x, alpha=0.02){
    return x < 0 ? alpha * x : x;
}

function leakyReluPrime(x, alpha=0.02){
    return x < 0 ? alpha : 1;
}

function elu(x, alpha=0.02){
    //https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu
    return x < 0 ? alpha * (Math.exp(x) - 1) : x;
}

function eluPrime(x, alpha=0.02){
    return x < 0 ? alpha * Math.exp(x) : 1;
}