class Layer{
    constructor(weightsNumber, neuronsNumber){
        this.weightsNumber = weightsNumber;
        this.neuronsNumber = neuronsNumber;
    }

    forward(){
        throw new Error('Missing impl');
    }

    backprop(){        
        throw new Error('Missing impl');
    }
}

if(typeof module !== 'undefined' && module.exports)
    module.exports = Layer;