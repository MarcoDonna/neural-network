class NeuralNetwork{
    constructor(layers){
        this.layers = layers;
        this.depth = layers.length;

        this.inputLayer = layers[0];
        this.outputLayer = layers[this.depth-1];
    }

    forward(inputs){
        //Input layer forward pass
        this.layers[0].forward(inputs);

        //Hidden and output layers forward pass
        for(let layerIndex = 1; layerIndex < this.depth; layerIndex++)
            this.layers[layerIndex].forward(this.layers[layerIndex-1]);
        return this.outputLayer.output;
    }
}