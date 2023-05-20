class RecurrentLayer extends Layer{
    constructor(weightsNumber, neuronsNumber, activationFunction, activationFunctionPrime, config={}){
        super(weightsNumber, neuronsNumber);

        this.activationFunction = activationFunction;
        this.activationFunctionPrime = activationFunctionPrime;

        this.gradientClipMagnitude = config.gradientClipMagnitude;
        
        this.regularization = config.regularization;
        this.regularizationRate = config.regularizationRate || 0.01;

        this.initRandomBiases();
        this.initRandomWeights();
        this.initRandomRecurrentWeights();

        this.initPartialBiasesDerivatives();
        this.initPartialWeightsDerivatives();
        this.initPartialRecurrentWeightsDerivatives();
    }

    get output(){
        return this.outputs;
    }

    get error(){
        return this.errors; 
    }

    initRandomBiases(){
        //Each neurons has one bias
        this.biases = [];
        for(let i = 0; i < this.neuronsNumber; i++)
            this.biases.push(Math.random());
    }

    initRandomWeights(){
        //Each neuron has one weight for each neuron in the layer-1
        this.weights = [];
        for(let i = 0; i < this.neuronsNumber; i++){
            this.weights.push([]);
            for(let j = 0; j < this.weightsNumber; j++)
                this.weights[i].push(Math.random());
        }
    }

    initRandomRecurrentWeights(){
        //Each neuron has one recurrent weight
        this.recurrentweights = [];
        for(let i = 0; i < this.neuronsNumber; i++)
            this.recurrentweights.push(-1 + 2 * Math.random());
    }
    
    initPartialBiasesDerivatives(){
        //Each bias, weight and recurrent weight has one partial
        //Same loop as above, with 0 instead of random
        this.biasesPartials = [];
        for(let i = 0; i < this.neuronsNumber; i++)
            this.biasesPartials.push(0);
    }

    initPartialWeightsDerivatives(){
        this.weightsPartials = [];
        for(let i = 0; i < this.neuronsNumber; i++){
            this.weightsPartials.push([]);
            for(let j = 0; j < this.weightsNumber; j++)
                this.weightsPartials[i].push(0);
        }
    }

    initPartialRecurrentWeightsDerivatives(){
        this.recurrentweightsPartials = [];
        for(let i = 0; i < this.neuronsNumber; i++)
            this.recurrentweightsPartials.push(0);
    }

    clearHiddenState(){
        this.hiddenstate = null;
        this.outputs = null;
    }

    forward(prevLayer){
        //Test input just like hidden layers
        if(!prevLayer.output || prevLayer.output.length != this.weightsNumber)
            throw new Error('Invalid input');

        this.inputs = prevLayer.output;

        /*
        The preactivation of a recurrent NEURON at t is
            inputs t . weights + hidden state t-1 * recurrent weight + bias
        same as hidden layer + hidden state t-1 * recurrent weight
        */
        
        //Update hidden state
        this.hiddenstate = [...this.outputs || []];

        this.activation = [];
        this.outputs = [];
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            const activation = dot(this.inputs, this.weights[neuronIndex]) + (this.hiddenstate[neuronIndex] || 0) * this.recurrentweights[neuronIndex] + this.biases[neuronIndex];
            this.activation.push(activation);
            this.outputs.push(this.activationFunction(activation));
        }
        
    }

    backprop(nextLayer){
        const nextLayerErrors = nextLayer.error;

        this.errors = [];
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            //The error is the same as dense layers
            const error = this.activationFunctionPrime(this.activation[neuronIndex]) * dot(nextLayerErrors, nextLayer.weightsToNeuron(neuronIndex));
            this.errors.push(error);

            //The gradient of the bias and the weights is the same as dense layer
            //The gradient of the recurrent weights, is the error of the neuron * hidden state (i think)
            this.biasesPartials[neuronIndex] += error;
            this.recurrentweightsPartials[neuronIndex] += error * (this.hiddenstate[neuronIndex] || 0);
            for(let weightIndex = 0; weightIndex < this.weightsNumber; weightIndex++)
                    this.weightsPartials[neuronIndex][weightIndex] += error * this.inputs[weightIndex];
        }
    }

    adjustBiases(learningRate, batchSize){
        //Just like dense
        //Here (and in all the adjust methods) batchsize is the amount of timesteps in training data
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            //Apply regularization
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
        //Just like dense
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            for(let weightIndex = 0; weightIndex < this.weightsNumber; weightIndex++){ 
                //Apply regularization
                let regularizationFactor = 0;
                if(this.regularization == 'l1')
                    regularizationFactor = Math.abs(this.weights[neuronIndex][weightIndex]) * this.regularizationRate;
                else if(this.regularization == 'l2')
                    regularizationFactor = 2 * this.weights[neuronIndex][weightIndex] * this.regularizationRate;

                const delta = -(learningRate/batchSize) * (this.weightsPartials[neuronIndex][weightIndex] + regularizationFactor);
                this.weights[neuronIndex][weightIndex] += delta;
            }
        }
        this.initPartialWeightsDerivatives();
    }

    adjustRecurrentWeights(learningRate, batchSize){
        //Really similar to biases in dense
        for(let neuronIndex = 0; neuronIndex < this.neuronsNumber; neuronIndex++){
            //Apply regularization
            let regularizationFactor = 0;
            if(this.regularization == 'l1')
                regularizationFactor = Math.abs(this.recurrentweights[neuronIndex]) * this.regularizationRate;
            else if(this.regularization == 'l2')
                regularizationFactor = 2 * this.recurrentweights[neuronIndex] * this.regularizationRate;

            const delta = -(learningRate/batchSize) * (this.recurrentweightsPartials[neuronIndex] + regularizationFactor);
            this.recurrentweights[neuronIndex] += delta;
        }
        this.initPartialRecurrentWeightsDerivatives();
    }
}