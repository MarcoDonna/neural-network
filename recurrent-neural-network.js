class RecurrentNeuralNetwork extends NeuralNetwork{
    constructor(layers){
        super(layers);
    }

    clearHiddenState(){
        for(let layerIndex = 0; layerIndex < this.depth; layerIndex++)
            if(this.layers[layerIndex].clearHiddenState)
                this.layers[layerIndex].clearHiddenState();
    }

    predict(inputsSeries){
        for(let recordIndex = 0; recordIndex < inputsSeries.length; recordIndex++){
            const inputs = inputsSeries[recordIndex];
            this.forward(inputs);
        }

        this.clearHiddenState();

        return this.outputLayer.output;
    }

    //Forward and backprop inherit form neural network

    train(features, targets, epochs, learningRate, lookbackAmount){
        for(let e = 0; e < epochs; e++)
            for(let currentTimeSeriesIndex = 0; currentTimeSeriesIndex < features.length; currentTimeSeriesIndex++)
                this.trainSingleTimeSeries(features[currentTimeSeriesIndex], targets[currentTimeSeriesIndex], learningRate, lookbackAmount);
    }

    trainSingleTimeSeries(timeSeriesFeatures, timeSeriesTargets, learningRate, lookbackAmount){
        /*
        Reason for the inner loop:
        This needs to be calced for each timestep, not over all the timeseries
        Can also be optimized to lookbac only on a certain amount of records in the series, improve training time and generalization
        */
        
        for(let recordIndex = 0; recordIndex < timeSeriesFeatures.length; recordIndex++){            
            const subsetTimeSeriesEnd = timeSeriesTargets.length - recordIndex;
            const subsetTimeSeriesStart = lookbackAmount ? Math.max(0, timeSeriesTargets.length - recordIndex - lookbackAmount) : 0;

            for(let partialRecordIndex = subsetTimeSeriesStart; partialRecordIndex < subsetTimeSeriesEnd; partialRecordIndex++){
                this.forward(timeSeriesFeatures[partialRecordIndex], true);
                this.backprop(timeSeriesTargets[partialRecordIndex]);
            }

            this.adjustLearnableParameters(learningRate, subsetTimeSeriesEnd - subsetTimeSeriesStart);
            this.clearHiddenState();
        }
    }

    adjustLearnableParameters(learningRate, batchSize){
        for(let layerIndex = 1; layerIndex < this.depth; layerIndex++)
            this.layers[layerIndex].adjustLearnableParameters(learningRate, batchSize);
    }
}