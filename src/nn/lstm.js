import { Module } from "./module.js";
import { LSTMCell } from "./lstm-cell.js";
import { Tensor } from "../core/tensor.js";

export class LSTM extends Module {
    constructor(inputSize, hiddenSize, bias = true) {
        super();

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // One shared cell across all timesteps.
        this.cell = new LSTMCell(inputSize, hiddenSize, bias);
    }

    forward(input, h0 = null, c0 = null) {
        const [seqLength, batchSize, inputSize] = input.shape;

        if (inputSize !== this.inputSize) {
            throw new Error(
                `Expected input size ${this.inputSize}, got ${inputSize}`
            );
        }

        // Initialize hidden state.
        let h = h0 ?? new Tensor(
            new Float32Array(batchSize * this.hiddenSize),
            [batchSize, this.hiddenSize],
            false
        );

        // Initialize cell state.
        let c = c0 ?? new Tensor(
            new Float32Array(batchSize * this.hiddenSize),
            [batchSize, this.hiddenSize],
            false
        );

        const outputs = [];

        for (let t = 0; t < seqLength; t++) {
            // Extract [batchSize, inputSize] for timestep t.
            const x = input.slice(`${t}:${t + 1}`, ":", ":").squeeze(0);

            // Advance both recurrent states.
            [h, c] = this.cell.forward(x, h, c);

            outputs.push(h);
        }

        // Handle empty sequences.
        if (seqLength === 0) {
            const output = new Tensor(
                new Float32Array(0),
                [0, batchSize, this.hiddenSize],
                false
            );

            return [output, h, c];
        }

        const output = Tensor.stack(outputs, 0);

        return [output, h, c];
    }
}
