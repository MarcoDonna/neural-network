class DenseLayer extends Layer{
    constructor(weightsNumber, neuronsNumber, activationFunction, activationFunctionPrime, config={}){
        super(weightsNumber, neuronsNumber);

        this.activationFunction = activationFunction;
        this.activationFunctionPrime = activationFunctionPrime;

        this.regularization = config.regularization;
        this.regularizationRate = config.regularizationRate || 0.01;

        this.initRandomBiases();
        this.initRandomWeights();

        this.initPartialBiasesDerivatives();
        this.initPartialWeightDerivatives();
    }

    get output(){
        return this.outputs;
    }

    get error(){
        return this.errors; 
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

    initPartialWeightDerivatives(){
        this.weightsPartials = [];
        for(let i = 0; i < this.neuronsNumber; i++){
            this.weightsPartials.push([]);
            for(let j = 0; j < this.weightsNumber; j++)
                this.weightsPartials[i].push(0);
        }
    }

    initPartialBiasesDerivatives(){
        this.biasesPartials = [];
        for(let i = 0; i < this.neuronsNumber; i++)
            this.biasesPartials.push(0);
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

    backprop(nextLayer){
        const nextLayerErrors = nextLayer.error;
        
        this.errors = [];
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            const error = this.activationFunctionPrime(this.activation[neuronIndex]) * dot(nextLayerErrors, nextLayer.weightsToNeuron(neuronIndex));
            this.errors.push(error);

            //Calc and store partial derivatives (both biases and weights)
            this.biasesPartials[neuronIndex] += error;
            for(let weightIndex = 0; weightIndex < this.weightsNumber; weightIndex++)
                this.weightsPartials[neuronIndex][weightIndex] += error * this.inputs[weightIndex];
        }
    }

    weightsToNeuron(prevNeuronIndex){
        return this.weights.map(neuron => neuron[prevNeuronIndex])
    }
    
    adjustLearnableParameters(learningRate, batchSize){
        this.adjustBiases(learningRate, batchSize);
        this.adjustWeights(learningRate, batchSize);
    }

    adjustBiases(learningRate, batchSize){
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            let regularizationFactor = 0;
            if(this.regularization == 'l1')
                regularizationFactor = Math.abs(this.biases[neuronIndex]) * this.regularizationRate;
            else if(this.regularization == 'l2')
                regularizationFactor = 2 * this.biases[neuronIndex] * this.regularizationRate;

            const delta = -(learningRate/batchSize) * (this.biasesPartials[neuronIndex] + regularizationFactor);
            this.biases[neuronIndex] += delta;
        }
        this.initPartialBiasesDerivatives();
    }

    adjustWeights(learningRate, batchSize){
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            for(let weightIndex = 0; weightIndex < this.weightsNumber; weightIndex++){ 
                let regularizationFactor = 0;
                if(this.regularization == 'l1')
                    regularizationFactor = Math.abs(this.weights[neuronIndex][weightIndex]) * this.regularizationRate;
                else if(this.regularization == 'l2')
                    regularizationFactor = 2 * this.weights[neuronIndex][weightIndex] * this.regularizationRate;

                const delta = -(learningRate/batchSize) * (this.weightsPartials[neuronIndex][weightIndex] + regularizationFactor);
                this.weights[neuronIndex][weightIndex] += delta;
            }
        }
        this.initPartialWeightDerivatives();
    }
}