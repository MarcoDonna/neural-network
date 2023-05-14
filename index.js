const {sigmoid, sigmoidPrime} = require('./activation');

const InputLayer = require('./input-layer');
const Input2DLayer = require('./input2d-layer');

const DenseLayer = require('./dense-layer');
const DropoutLayer = require('./dropout-layer');
const ConvolutionalLayer = require('./convolutional-layer');

const OutputLayer = require('./output-layer');

const NeuralNetwork = require('./neural-network');

const xor_f = [[0, 0], [0, 1], [1, 0], [1, 1]];
const xor_t = [[1, 0], [0, 1], [0, 1], [1, 0]];

const layers = [
    new InputLayer(2),
    new DenseLayer(2, 4, sigmoid, sigmoidPrime),
    new OutputLayer(4, 2, sigmoid, sigmoidPrime)
];

const nn = new NeuralNetwork(layers);

nn.train(xor_f, xor_t, 10000, 0.5, 2);

console.log(nn.predict([0, 0]));
console.log(nn.predict([0, 1]));
console.log(nn.predict([1, 0]));
console.log(nn.predict([1, 1]));