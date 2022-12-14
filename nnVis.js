const networkDebugger = (nn, canvas, parameters) => {
    const {LEFT_MARGIN, TOP_MARGIN, NEURON_RADIUS, NEURON_PADDING_X, NEURON_PADDING_Y} = parameters;
    canvas.font = parameters.font || "10px Arial";

    for(let i = 0; i < nn.layers.length; i++)
        for(let j = 0; j < nn.layers[i].neurons.length; j++){
            const neuron = nn.layers[i].neurons[j];
            const left = LEFT_MARGIN + i * (NEURON_PADDING_X + NEURON_RADIUS);
            const top = TOP_MARGIN + j * (NEURON_PADDING_Y + NEURON_RADIUS);

            
            canvas.beginPath();
            canvas.strokeStyle = `rgb(0, 0, 0)`;
            canvas.arc(left, top, NEURON_RADIUS, 1/2*Math.PI, 3/2*Math.PI);
            canvas.arc(left, top, NEURON_RADIUS, 1/2*Math.PI, 3/2*Math.PI, true);
            canvas.stroke();

            const red = 255 * (neuron.weights[0] + 1) / 2;
            const blue = 255 - 255 * (neuron.weights[0] + 1) / 2;
            
            canvas.fillStyle = `rgb(${red}, 0, ${blue})`;
            canvas.fillText(neuron.weights[0].toFixed(3), left-NEURON_RADIUS+5, top-5);
            canvas.fillStyle = `rgb(0, 0, 0)`;
            if(neuron.h !== undefined && neuron.o !== undefined){
                canvas.fillText(neuron.h.toFixed(3), left-NEURON_RADIUS+5, top+5);
                canvas.fillText(neuron.o.toFixed(3), left+5, top);
                if(nn.layers[i].error !== undefined)
                    canvas.fillText(nn.layers[i].error[j].toFixed(6), left-20, top+NEURON_RADIUS+10)
            }
            
            if(i != 0){
                for(let k = 0; k < nn.layers[i-1].neurons.length; k++){
                    //Loop each neuron in prev layer
                    canvas.beginPath();
                    const red = 255 * (neuron.weights[k+1] + 1) / 2;
                    const blue = 255 - 255 * (neuron.weights[k+1] + 1) / 2;
                    canvas.moveTo(left - NEURON_RADIUS, top);
                    canvas.lineTo(LEFT_MARGIN + (i-1) * (NEURON_PADDING_X + NEURON_RADIUS) + NEURON_RADIUS, TOP_MARGIN + k * (NEURON_PADDING_Y + NEURON_RADIUS));
                    canvas.strokeStyle = `rgb(${red}, 0, ${blue})`;
                    canvas.stroke();
                }
            }

        }
}