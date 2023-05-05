class Layer{
    constructor(weightsNumber, neuronsNumber){
        this.weightsNumber = weightsNumber;
        this.neuronsNumber = neuronsNumber;
    }

    forward(){
        this.errorMissingImplementation();
    }

    backprop(){        
        this.errorMissingImplementation();
    }

    #errorMissingImplementation(){
        throw new Error('Missing impl');
    }
}