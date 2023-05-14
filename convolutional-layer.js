class ConvolutionalLayer{
    constructor(inputWidth, inputHeigth, activationFunction, activationFunctionPrime, config={}){
        this.inputWidth = inputWidth;
        this.inputHeigth = inputHeigth;
        
        this.activationFunction = activationFunction;
        this.activationFunctionPrime = activationFunctionPrime;

        this.filter = {};
        this.filter.size = config.size || 5;
        this.filter.stride = config.stride || 1;
        //this.flatten = config.flatten || true;

        this.filter.numberHeigth = (this.inputWidth - this.filter.size) / this.filter.stride + 1;
        this.filter.numberWidth = (this.inputHeigth - this.filter.size) / this.filter.stride + 1;
        this.filter.number = this.filter.numberHeigth * this.filter.numberWidth;
    
        this.initRandomBiases();
        this.initRandomWeights();
    }

    get output(){
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

    forward(prevLayer){
        if(!prevLayer.output || prevLayer.output.length != this.inputHeigth || prevLayer.output[0].length != this.inputWidth)
            throw new Error('Invalid input');

        this.inputs = prevLayer.output;

        this.activation = [];
        this.outputs = [];

        for(let filterRowIndex = 0; filterRowIndex < this.filter.numberHeigth; filterRowIndex++){
            this.activation.push([]);
            this.outputs.push([]);
            for(let filterColIndex = 0; filterColIndex < this.filter.numberWidth; filterColIndex++){
                //Apply filter to input, store result in activation and output
                let activation = this.biases[filterRowIndex][filterColIndex];

                const filterStartRow = filterRowIndex * this.filter.stride;
                const filterStartCol = filterColIndex * this.filter.stride;

                for(let i = 0; i < this.filter.size; i++){
                    const inputsSubset = this.inputs[filterStartRow].slice(filterStartCol, filterStartCol + this.filter.size);
                    activation += dot(inputsSubset, this.weights[filterRowIndex][filterColIndex][i]);
                }                

                this.activation[filterRowIndex].push(activation);
                this.outputs[filterRowIndex].push(this.activationFunction(activation));
            }
        }        
    }
}