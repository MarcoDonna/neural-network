const networkDebugger = (nn, canvas, parameters) => {
    const {LEFT_MARGIN, TOP_MARGIN, NEURON_RADIUS, NEURON_PADDING_X, NEURON_PADDING_Y} = parameters;
    
    let context = canvas.getContext("2d");
    context.font = `${parameters.fontSize}px ${parameters.font}` || "10px Arial";
    context.clearRect(0, 0, canvas.width, canvas.height);

    for(let i = 0; i < nn.layers.length; i++)
        for(let j = 0; j < nn.layers[i].neurons.length; j++){
            const neuron = nn.layers[i].neurons[j];
            const left = LEFT_MARGIN + i * (NEURON_PADDING_X + NEURON_RADIUS);
            const top = TOP_MARGIN + j * (NEURON_PADDING_Y + NEURON_RADIUS);

            
            context.beginPath();
            context.strokeStyle = `rgb(0, 0, 0)`;
            context.arc(left, top, NEURON_RADIUS, 1/2*Math.PI, 3/2*Math.PI);
            context.arc(left, top, NEURON_RADIUS, 1/2*Math.PI, 3/2*Math.PI, true);
            context.stroke();

            const red = 255 * (neuron.weights[0] + 1) / 2;
            const blue = 255 - 255 * (neuron.weights[0] + 1) / 2;
            
            context.fillStyle = `rgb(${red}, 0, ${blue})`;
            context.fillText(neuron.weights[0].toFixed(3), left-NEURON_RADIUS+5, top-5);
            context.fillStyle = `rgb(0, 0, 0)`;
            if(neuron.h !== undefined && neuron.o !== undefined){
                context.fillText(neuron.h.toFixed(3), left-NEURON_RADIUS+5, top+5);
                context.fillText(neuron.o.toFixed(3), left+5, top);
                if(nn.layers[i].error !== undefined)
                    context.fillText(nn.layers[i].error[j].toFixed(6), left-20, top+NEURON_RADIUS+10)
            }
            
            if(i != 0){
                for(let k = 0; k < nn.layers[i-1].neurons.length; k++){
                    //Loop each neuron in prev layer
                    context.beginPath();
                    const red = 255 * (neuron.weights[k+1] + 1) / 2;
                    const blue = 255 - 255 * (neuron.weights[k+1] + 1) / 2;
                    context.moveTo(left - NEURON_RADIUS, top);
                    context.lineTo(LEFT_MARGIN + (i-1) * (NEURON_PADDING_X + NEURON_RADIUS) + NEURON_RADIUS, TOP_MARGIN + k * (NEURON_PADDING_Y + NEURON_RADIUS));
                    context.strokeStyle = `rgb(${red}, 0, ${blue})`;
                    context.stroke();
                }
            }

        }
}