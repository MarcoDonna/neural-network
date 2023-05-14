class NeuralNetwork{
    constructor(layers){
        this.layers = layers;
        this.depth = layers.length;

        this.inputLayer = layers[0];
        this.outputLayer = layers[this.depth-1];
    }

    predict(inputs){
        return this.forward(inputs, false);
    }

    forward(inputs, training){
        //Input layer forward pass
        this.inputLayer.forward(inputs);

        //Hidden and output layers forward pass
        for(let layerIndex = 1; layerIndex < this.depth; layerIndex++)
            this.layers[layerIndex].forward(this.layers[layerIndex-1], training);
        return this.outputLayer.output;
    }

    backprop(targets){
        //Output layer error
        this.outputLayer.backprop(targets);

        //Backprop error in hidden layers
        for(let layerIndex = this.depth - 2; layerIndex > 0; layerIndex--)
            this.layers[layerIndex].backprop(this.layers[layerIndex+1]);
    }

    train(features, targets, epochs, learningRate, batchSize){
        //Split data in batches
        const batchesFeatures = [];
        const batchesTargets = [];

        const batchNum = Math.ceil(features.length/batchSize);

        for(let i = 0; i < features.length; i += batchNum){
            batchesFeatures.push(features.slice(i, i + batchNum));
            batchesTargets.push(targets.slice(i, i + batchNum));
        }

        //Train model
        for(let e = 0; e <= epochs; e++){
            if(e % 1000 == 0)
                console.log(`${e}/${epochs}`);
            for(let i = 0; i < batchesFeatures.length; i++)
                this.trainSingleBatch(batchesFeatures[i], batchesTargets[i], learningRate, batchSize);
        }
    }

    trainSingleBatch(features, targets, learningRate){
        //Do a forward and backprop pass for each X, Y pari in batch
        for(let recordIndex = 0; recordIndex < features.length; recordIndex++){
            this.forward(features[recordIndex], true /*true when during training */);
            this.backprop(targets[recordIndex]);
        }
        this.adjustBiases(learningRate, features.length);
        this.adjustWeights(learningRate, features.length);
    }

    adjustWeights(learningRate, batchSize){
        for(let layerIndex = 1; layerIndex < this.depth; layerIndex++)
            this.layers[layerIndex].adjustWeights(learningRate, batchSize);
    }

    adjustBiases(learningRate, batchSize){
        for(let layerIndex = 1; layerIndex < this.depth; layerIndex++)
            this.layers[layerIndex].adjustBiases(learningRate, batchSize);
    }
}

if(typeof module !== 'undefined' && module.exports)
    module.exports = NeuralNetwork;