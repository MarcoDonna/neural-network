class DenseLayer extends Layer{
    constructor(weightsNumber, neuronsNumber, activationFunction, activationFunctionPrime){
        super(weightsNumber, neuronsNumber);

        this.activationFunction = activationFunction;
        this.activationFunctionPrime = activationFunctionPrime;

        this.initRandomBiases();
        this.initRandomWeights();
    }

    get output(){
        return this.outputs;
    }

    initRandomWeights(){
        this.weights = [];
        for(let i = 0; i < this.neuronsNumber; i++){
            this.weights.push([]);
            for(let j = 0; j < this.weightsNumber; j++)
                this.weights[i].push(Math.random());
        }
    }

    initRandomBiases(){
        this.biases = [];
        for(let i = 0; i < this.neuronsNumber; i++)
            this.biases.push(Math.random());
    }

    forward(prevLayer){
        if(!prevLayer.output || prevLayer.output.length != this.weightsNumber)
            throw new Error('Invalid input');
        
        this.inputs = prevLayer.output;
        
        this.activation = [];
        this.outputs = [];
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            const activation = dot(this.inputs, this.weights[neuronIndex]) + this.biases[neuronIndex];
            this.activation.push(activation);
            this.outputs.push(this.activationFunction(activation));
        }
    }
}