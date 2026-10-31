import { Tensor } from "../core/tensor.js";
import { GRUCell } from "./gru-cell.js";
import { Module } from "./module.js";

export class GRU extends Module {
    constructor(inputSize, hiddenSize, bias = true) {
        super();

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.cell = new GRUCell(
            inputSize,
            hiddenSize,
            bias
        );
    }

    forward(input, h0 = null) {
        const [seq, batch, inputSize] = input.shape;

        // Validate input
        if (input.shape.length !== 3) {
            throw new Error(
                "GRU input must have shape [seq, batch, inputSize]"
            );
        }

        if (inputSize !== this.inputSize) {
            throw new Error("GRU input size mismatch");
        }

        // Initialize hidden state
        let h = h0 ?? new Tensor(
            new Float32Array(batch * this.hiddenSize),
            [batch, this.hiddenSize],
            false
        );

        const outputs = [];

        // Process sequence
        for (let t = 0; t < seq; t++) {
            const x = input
                .slice(`${t}:${t + 1}`, ":", ":")
                .squeeze(0);

            h = this.cell.forward(x, h);

            outputs.push(h);
        }

        // Handle empty sequence
        if (seq === 0) {
            const output = new Tensor(
                new Float32Array(0),
                [0, batch, this.hiddenSize],
                false
            );

            return [output, h];
        }

        return [
            Tensor.stack(outputs, 0),
            h
        ];
    }
}