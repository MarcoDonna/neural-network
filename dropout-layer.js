class DropoutLayer extends Layer{
    constructor(weightsNumber, neuronsNumber, activationFunction, activationFunctionPrime, config={}){
        super(weightsNumber, neuronsNumber);

        this.activationFunction = activationFunction;
        this.activationFunctionPrime = activationFunctionPrime;
        this.dropoutRate = config.dropoutRate || 0.2

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

    initRandomMask(){
        this.mask = [];
        for(let i = 0; i < this.neuronsNumber; i++)
            this.mask.push(Math.random() < this.dropoutRate ? 0 : 1);
    }

    applyMask(vector){
        if(vector.length != this.neuronsNumber)
            throw new Error('Invalid mask or input');

        for(let i = 0; i < vector.length; i++)
            vector[i] *= this.mask[i];
    }

    forward(prevLayer, training){
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

        //Apply dropout mask if training
        if(training == true){
            this.initRandomMask();
            this.applyMask(this.activation);
            this.applyMask(this.outputs);
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

        //Apply dropout mask (backprop is only done when training)
        this.applyMask(this.errors);
    }

    weightsToNeuron(prevNeuronIndex){
        return this.weights.map(neuron => neuron[prevNeuronIndex])
    }

    adjustBiases(learningRate, batchSize){
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            let regularizationFactor = 0;
            if(this.regularization == 'l1')
                regularizationFactor = Math.sign(this.biases[neuronIndex]) * this.regularizationRate;
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
                    regularizationFactor = Math.sign(this.weights[neuronIndex][weightIndex]) * this.regularizationRate;
                else if(this.regularization == 'l2')
                    regularizationFactor = 2 * this.weights[neuronIndex][weightIndex] * this.regularizationRate;

                const delta = -(learningRate/batchSize) * (this.weightsPartials[neuronIndex][weightIndex] + regularizationFactor);
                this.weights[neuronIndex][weightIndex] += delta;
            }
        }
        this.initPartialWeightDerivatives();
    }
}