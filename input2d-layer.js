class Input2DLayer{
    constructor(heigth, width){
        this.width = width;
        this.heigth = heigth;
    }

    get output(){
        return this.outputs;
    }

    forward(inputs){
        if(inputs.length != this.heigth || inputs[0].length != this.width)
            throw new Error('Invalid input size');
        this.outputs = inputs;
    }
}

if(typeof module !== 'undefined' && module.exports)
    module.exports = Input2DLayer;