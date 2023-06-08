class ConvolutionalLayer{
    constructor(inputWidth, inputHeigth, activationFunction, activationFunctionPrime, config={}){
        this.inputWidth = inputWidth;
        this.inputHeigth = inputHeigth;
        
        this.activationFunction = activationFunction;
        this.activationFunctionPrime = activationFunctionPrime;

        this.filter = {};
        this.filter.size = config.size || 5;
        this.filter.stride = config.stride || 1;
        
        this.flatten = config.flatten || true;

        this.filter.numberHeigth = (this.inputWidth - this.filter.size) / this.filter.stride + 1;
        this.filter.numberWidth = (this.inputHeigth - this.filter.size) / this.filter.stride + 1;
        this.filter.number = this.filter.numberHeigth * this.filter.numberWidth;
    
        this.initRandomBiases();
        this.initRandomWeights();

        this.initPartialBiasesDerivatives();
        this.initPartialWeightDerivatives();    
    }

    get output(){
        if(this.flatten == true)
            return this.outputFlat;
        return this.outputs;
    }

    get outputFlat(){
        let ret = [];
        for(let i = 0; i < this.outputs.length; i++)
            ret.push(...this.outputs[i]);
        return ret;
    }

    get error(){
        return this.errors;
    }

    initRandomBiases(){
        this.biases = [];
        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++){
            this.biases.push([]);
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++)
                this.biases[filterRowIndex].push(Math.random());
        }
    }

    initRandomWeights(){
        this.weights = [];
        
        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++){
            this.weights.push([]);
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++){
                this.weights[filterRowIndex].push([]);
                for(let i = 0; i < this.filter.size; i++){
                    this.weights[filterRowIndex][filterColIndex].push([]);
                    for(let j = 0; j < this.filter.size; j++)
                        this.weights[filterRowIndex][filterColIndex][i].push(Math.random());
                }
            }
        }
    }

    initPartialBiasesDerivatives(){
        this.biasesPartials = [];

        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++){
            this.biasesPartials.push([]);
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++){
                this.biasesPartials[filterRowIndex].push(0);
            }
        }
    }

    initPartialWeightDerivatives(){
        this.weightsPartials = [];

        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++){
            this.weightsPartials.push([]);
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++){
                this.weightsPartials[filterRowIndex].push([]);
                for(let i = 0; i < this.filter.size; i++){
                    this.weightsPartials[filterRowIndex][filterColIndex].push([]);
                    for(let j = 0; j < this.filter.size; j++)
                        this.weightsPartials[filterRowIndex][filterColIndex][i].push(0);
                }
            }
        }
    }

    forward(prevLayer){
        if(!prevLayer.output || prevLayer.output.length != this.inputHeigth || prevLayer.output[0].length != this.inputWidth)
            throw new Error('Invalid input');

        this.inputs = prevLayer.output;

        this.activation = [];
        this.outputs = [];

        //For each filter
        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++){
            this.activation.push([]);
            this.outputs.push([]);
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++){
                //Apply filter to input, store result in activation and output
                let activation = this.biases[filterRowIndex][filterColIndex];

                const filterStartRow = filterRowIndex * this.filter.stride;
                const filterStartCol = filterColIndex * this.filter.stride;

                for(let i = 0; i < this.filter.size; i++){
                    const inputSubset = this.inputs[filterStartRow+i].slice(filterStartCol, filterStartCol + this.filter.size);
                    activation += dot(inputSubset, this.weights[filterRowIndex][filterColIndex][i]);
                }          

                this.activation[filterRowIndex].push(activation);
                this.outputs[filterRowIndex].push(this.activationFunction(activation));
            }
        }        
    }

    backprop(nextLayer){
        const nextLayerErrors = nextLayer.errors;

        this.errors = [];
        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++){
            this.errors.push([]);
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++){
                //each filter can be considered a neuron, need to "de-flatten" error
                //row * size + col
                const weightsToNeuron = nextLayer.weightsToNeuron(filterRowIndex * this.filter.size + filterColIndex);
                const error = this.activationFunctionPrime(this.activation[filterRowIndex][filterColIndex]) * dot(nextLayerErrors, weightsToNeuron);
                this.errors[filterRowIndex].push(error);

                //calc gradients
                this.biasesPartials[filterRowIndex][filterColIndex] += error;
                for(let i = 0; i < this.filter.size; i++)
                    for(let j = 0; j < this.filter.size; j++){
                        const inputRow = filterRowIndex * this.filter.stride + i;
                        const inputCol = filterColIndex * this.filter.stride + j;
                        const input = this.inputs[inputRow][inputCol];
                        this.weightsPartials[filterRowIndex][filterColIndex][i][j] += error * input;
                    }
            }
        }
    }

    adjustLearnableParameters(learningRate, batchSize){
        this.adjustBiases(learningRate, batchSize);
        this.adjustWeights(learningRate, batchSize);
    }

    adjustBiases(learningRate, batchSize){
        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++)
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++){
                const delta = -(learningRate/batchSize) * this.biasesPartials[filterRowIndex][filterColIndex];
                this.biases[filterRowIndex][filterColIndex] += delta;
            }
        this.initPartialBiasesDerivatives();
    }

    adjustWeights(learningRate, batchSize){
        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++)
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++){
                
                for(let i = 0; i < this.filter.size; i++)
                    for(let j = 0; j < this.filter.size; j++){
                        const delta = -(learningRate/batchSize) * this.weightsPartials[filterRowIndex][filterColIndex][i][j];
                        this.weights[filterRowIndex][filterColIndex][i][j] += delta;
                    }
            }
        this.initPartialWeightDerivatives();
    }
}