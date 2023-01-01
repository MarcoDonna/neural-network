class DeepAgent{
    constructor(stateSize, actions, config, data){
        this.config = config;

        this.action = actions;

        this.main = new NeuralNetwork();

        this.data = data;   //Agent custom parameters

        this.replay = [];
    }

    maxAction(state){
        //ArgMax(QValues@state)
        let prediction = this.main.forward(state).out;
        let maxAction = prediction[0], maxActionIndex = 0;
        for(let i = 0; i < prediction.length; i++)
            if(prediction[i] > maxAction){
                maxAction = prediction[i];
                maxActionIndex = i;
            }

        return {i: maxActionIndex, a: this.action[maxActionIndex], q: maxAction};
    }

    randAction(state){
        let randIndex = Math.floor(Math.random() * this.action.length);
        return {i: randIndex, a: this.action[randIndex]};
    }

    train(){
        for(let e = 1; e <= this.config.episodes; e++){
            let state = this.state();
            let action = this.maxAction(state);

            if(Math.random() < (typeof this.config.epsilon === "function") ? this.config.epsilon(e) : this.config.epsilon)
                //Epsilon greedy explore-exploit tradeoff, can be a fixed value or a function with epoch parameter 
                action = this.randAction();
                
            let nextState = this.nextState(state, action);
            let reward = this.reward(nextState);
            let done = this.isFinalState(nextState);

            //Store experience in experience memory
            this.replay.push({state, action, reward, nextState, done});

            //Reset agent/environment if final or if reeached episode length
            if(done || this.config.episodeLength && e%this.config.episodeLength==0)
                this.reset();
            else
                this.state(nextState);

            if(e%this.config.replayMemSize==0){
                let stateBatch = [], predictionBatch = [];

                for(let i = 0; i < this.config.replayMemSize; i++){
                    //Random Sampling, shuffle replay data and store state and targets (updated predictions)
                    let randIdx = Math.floor(Math.random() * this.replay.length);
                    let {state, action, reward, nextState, done} = this.replay[randIdx];//this.replay[Math.floor(Math.random() * this.replay.length)];
                    this.replay.splice(randIdx,1);
                    let maxAction = this.maxAction(nextState);
                    let prediction = this.main.forward(state).out;
                    if(done)
                        prediction[action.i] = reward;
                    else
                        prediction[action.i] = reward + this.config.gamma * maxAction.q;

                        stateBatch.push(state);
                        predictionBatch.push(prediction);

                }
                //Train nn
                this.main.train(stateBatch, predictionBatch, this.config.epochs, this.config.alpha, this.config.batchSize);
                this.replay.mem = [];
            }                    
        }
        return this;
    }
}