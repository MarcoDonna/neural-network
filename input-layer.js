const {dot} = require('./utils');
const Layer = require('./layer');

class InputLayer extends Layer{
    constructor(neuronsNumber){
        super(null, neuronsNumber);
    }

    get output(){
        return this.outputs;
    }

    forward(inputs){
        if(inputs.length != this.neuronsNumber)
            throw new Error('Invalid input size');
        this.outputs = inputs;
    }
}

if(typeof module !== 'undefined' && module.exports)
    module.exports = InputLayer;