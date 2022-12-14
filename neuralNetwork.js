class Neuron{
    constructor(inputSize){
        this.weightsError = []; //Used to calc batch error
        this.weights = [Math.random()]; //Weights[0] is bias
        for(let i = 0; i < inputSize; i++)
            this.weights.push(Math.random() * 2 - 1); //One weight for each input
    }

    forward(inputVector, activation){
        this.h = this.weights[0];
        for(let i = 1; i < this.weights.length; i++)
            this.h += inputVector[i-1] * this.weights[i]; //Product sum plus bias
        this.o = activation.f(this.h); //Node output
        return this;
    }
}

class Layer{
    //Dense layer
    constructor(neurons, inputs, activation){
        this.neurons = [];
        this.activation = activation;

        for(let i = 0; i < neurons; i++)
            this.neurons.push(new Neuron(inputs));
    }

    forward(input){
        for(let i = 0; i < this.neurons.length; i++)
            this.neurons[i].forward(input, this.activation);
    }
}

class OutputLayer extends Layer{
    constructor(neurons, inputs, activation){
        super(neurons, inputs, activation);
    }
}

class InputLayer extends Layer{
    constructor(inputs){
        super(inputs, 1, linear);

        for(let i = 0; i < this.neurons.length; i++)
            for(let j = 0; j < this.neurons[i].weights.length; j++)
                //Initialize weight to 1, bias to 0 with linear activation function to pass inputs without any transformation
                if(j == 0)
                    this.neurons[i].weights[j] = 0;
                else
                    this.neurons[i].weights[j] = 1;
    }

    forward(inputs){
        for(let i = 0; i < this.neurons.length; i++)
            this.neurons[i].forward([inputs[i]], this.activation);
    }
}

class NeuralNetwork{
    constructor(shape){
        this.layers = [];

        //Build network
        this.layers.push(new InputLayer(shape[0]));
        for(let i = 1; i < shape.length-1; i++)
            this.layers.push(new Layer(shape[i], shape[i-1], relu))
        this.layers.push(new OutputLayer(shape[shape.length-1], shape[shape.length-2], sigmoid));                    
    }

    forward(inputVector){
        let layerOutput = inputVector;
        for(let i = 0; i < this.layers.length; i++){
            //Prev layer output becomes input
            this.layers[i].forward(layerOutput);
            layerOutput = this.layers[i].neurons.map(neuron => neuron.o);                    
        }
        //network.out => output layer prediction
        this.out = layerOutput;
        return this;
    }

    batchTrain(predictors, classes, learningRate){
        for(let recordIdx = 0; recordIdx < predictors.length; recordIdx++){
            //Feedforward
            this.forward(predictors[recordIdx]);

            //Output layer error
            const layer = this.layers[this.layers.length-1]
            layer.error = [];                    
            for(let nrnIdx = 0; nrnIdx < layer.neurons.length; nrnIdx++){
                const neuron = this.layers[this.layers.length-1].neurons[nrnIdx];
                //console.log(`${layer.activation.d(neuron.h)} * (${this.out[nrnIdx]} - ${classes[recordIdx][nrnIdx]})`)
                layer.error.push(layer.activation.d(neuron.h) * (this.out[nrnIdx] - classes[recordIdx][nrnIdx]));
            }

            //Hidden layer error
            for(let lyrIdx = this.layers.length-2; lyrIdx > 0; lyrIdx--){
                const layer = this.layers[lyrIdx];
                const nextLayer = this.layers[lyrIdx+1];
                layer.error = [];

                for(let nrnIdx = 0; nrnIdx < layer.neurons.length; nrnIdx++){
                    const neuron = layer.neurons[nrnIdx];
                    
                    let err = layer.activation.d(neuron.h)
                    let sum = 0;

                    for(let i = 0; i < nextLayer.neurons.length; i++){  
                        sum += nextLayer.neurons[i].weights[nrnIdx + 1] * nextLayer.error[i];
                    }
                    layer.error.push(err * sum);
                }
            }

            //Partial Error
            for(let lyrIdx = 1; lyrIdx < this.layers.length; lyrIdx++){
                const layer = this.layers[lyrIdx];
                const prevLayer = this.layers[lyrIdx-1];
                for(let nrnIdx = 0; nrnIdx < layer.neurons.length; nrnIdx++){
                    const neuron = layer.neurons[nrnIdx];
                    for(let wghIdx = 0; wghIdx < neuron.weights.length; wghIdx++){
                        if(!neuron.weightsError[wghIdx])
                            neuron.weightsError.push(0);
                        if(wghIdx == 0)
                            neuron.weightsError[wghIdx] += layer.error[nrnIdx];
                        else
                            neuron.weightsError[wghIdx] += layer.error[nrnIdx] * prevLayer.neurons[wghIdx-1].o;
                    }
                }
            }
        }

        //Combine partial errors (avg) and adjust weights
        for(let lyrIdx = 1; lyrIdx < this.layers.length; lyrIdx++){
            const layer = this.layers[lyrIdx];
            for(let nrnIdx = 0; nrnIdx < layer.neurons.length; nrnIdx++){
                const neuron = layer.neurons[nrnIdx];
                for(let wghIdx = 0; wghIdx < neuron.weights.length; wghIdx++){
                    neuron.weights[wghIdx] -= learningRate * (neuron.weightsError[wghIdx] / classes.length);                            
                }
                neuron.weightsError = []
            }
        }

        return this;
    }

    train(predictors, classes, epochs, learningRate, batchSize){
        let batch = [];

        //Calc num of batches
        let batchNum = Math.ceil(predictors.length/batchSize);

        for(let i = 0; i < predictors.length; i += batchNum){
            //Split data in batches
            batch.push({
                predictors: predictors.slice(i, i + batchNum),
                classes: classes.slice(i, i + batchNum)
            });
        }

        for(let e = 0; e < epochs; e++)
            for(let i = 0; i < batch.length;  i++)
                this.batchTrain(batch[i].predictors, batch[i].classes, learningRate);
        
        return this;
    }
}