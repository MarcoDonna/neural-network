class ConvolutionalLayer{
    constructor(inputWidth, inputHeigth, activationFunction, activationFunctionPrime, config={}){
        this.activationFunction = activationFunction;
        this.activationFunctionPrime = activationFunctionPrime;

        this.filter = {};
        this.filter.size = config.size || 5;
        this.filter.stride = config.stride || 1;
        //this.flatten = config.flatten || true;

        this.filter.numberHeigth = (inputHeigth - this.filter.size) / this.filter.stride + 1;
        this.filter.numberWidth = (inputWidth - this.filter.size) / this.filter.stride + 1;
        this.filter.number = this.filter.numberHeigth * this.filter.numberWidth;
    
        this.initRandomBiases();
        this.initRandomWeights();
    }

    get output(){
        return this.outputs;
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
}