import { Tensor } from "../core/tensor.js";
import { Module } from "./module.js";
import { RNNCell } from "./rnn-cell.js";


export class RNN extends Module {
    constructor(inputSize, hiddenSize) {
        super();

        if (!Number.isInteger(inputSize) || inputSize <= 0) {
            throw new Error("inputSize must be a positive integer");
        }

        if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
            throw new Error("hiddenSize must be a positive integer");
        }

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.cell = new RNNCell(inputSize, hiddenSize);
    }

    forward(input, h0 = null) {
        // Validate input
        if (!(input instanceof Tensor)) {
            throw new Error("Expected input to be a Tensor");
        }

        if (input.shape.length !== 3) {
            throw new Error(
                "RNN input must have shape [seq, batch, input]"
            );
        }

        const [seqLength, batchSize, inputSize] = input.shape;

        if (inputSize !== this.inputSize) {
            throw new Error(
                `Expected input size ${this.inputSize}, got ${inputSize}`
            );
        }

        // Validate initial hidden state
        if (h0 !== null) {
            if (!(h0 instanceof Tensor)) {
                throw new Error("Expected h0 to be a Tensor");
            }

            if (
                h0.shape.length !== 2 ||
                h0.shape[0] !== batchSize ||
                h0.shape[1] !== this.hiddenSize
            ) {
                throw new Error(
                    `Expected h0 shape [${batchSize}, ${this.hiddenSize}]`
                );
            }
        }

        // Initialize hidden state
        let h = h0 ?? new Tensor(
            new Float32Array(batchSize * this.hiddenSize),
            [batchSize, this.hiddenSize]
        );

        // Handle empty sequence
        if (seqLength === 0) {
            return [
                new Tensor(
                    new Float32Array(0),
                    [0, batchSize, this.hiddenSize]
                ),
                h
            ];
        }

        const outputs = [];

        for (let t = 0; t < seqLength; t++) {

            const sliced = input.slice(
                `${t}:${t + 1}`,
                ":",
                ":"
            );

            const x_t = sliced.squeeze(0);
            h = this.cell.forward(x_t, h);

            outputs.push(h);
        }

        const output = Tensor.stack(outputs, 0);
        return [
            output,
            h
        ];
    }
}