class NeuralNetwork{
    constructor(layers=[]){
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
        for(let e = 0; e < epochs; e++)
            for(let i = 0; i < batchesFeatures.length; i++)
                this.trainSingleBatch(batchesFeatures[i], batchesTargets[i], learningRate, batchSize);
    }

    trainSingleBatch(features, targets, learningRate){
        //Do a forward and backprop pass for each X, Y pari in batch
        for(let recordIndex = 0; recordIndex < features.length; recordIndex++){
            this.forward(features[recordIndex], true /*true when during training */);
            this.backprop(targets[recordIndex]);
        }
        this.adjustLearnableParameters(learningRate, features.length);
    }

    adjustLearnableParameters(learningRate, batchSize){
        for(let layerIndex = 1; layerIndex < this.depth; layerIndex++)
            this.layers[layerIndex].adjustLearnableParameters(learningRate, batchSize);
    }

    export(){
        return {
            layers: this.layers,
            depth: this.depth
        }
    }

    import(data){
        this.layers = data.layers;
        this.depth = data.depth;
        this.inputLayer = this.layers[0];
        this.outputLayer = this.layers[this.depth-1];
    }

    save(){
        return JSON.stringify(this.export());
    }

    load(json){
        this.import(JSON.parse(json));
    }
}