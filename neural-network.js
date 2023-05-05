class NeuralNetwork{
    constructor(layers){
        this.layers = layers;
        this.depth = layers.length;

        this.inputLayer = layers[0];
        this.outputLayer = layers[this.depth-1];
    }

    forward(inputs){
        //Input layer forward pass
        this.inputLayer.forward(inputs);

        //Hidden and output layers forward pass
        for(let layerIndex = 1; layerIndex < this.depth; layerIndex++)
            this.layers[layerIndex].forward(this.layers[layerIndex-1]);
        return this.outputLayer.output;
    }

    backprop(targets){
        this.outputLayer.backprop(targets);
        for(let layerIndex = this.depth - 2; layerIndex > 0; layerIndex--)
            this.layers[layerIndex].backprop(this.layers[layerIndex+1]);
    }
}