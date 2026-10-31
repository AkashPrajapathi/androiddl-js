import { Linear } from "./linear.js";
import { Module } from "./module.js";
import { Sigmoid } from "./sigmoid.js";
import { Tanh } from "./tanh.js";

export class LSTMCell extends Module {
    constructor(inputSize, hiddenSize, bias = true) {
        super();

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // Forget gate
        this.xf = new Linear(inputSize, hiddenSize, bias);
        this.hf = new Linear(hiddenSize, hiddenSize, bias);

        // Input gate
        this.xi = new Linear(inputSize, hiddenSize, bias);
        this.hi = new Linear(hiddenSize, hiddenSize, bias);

        // Candidate cell
        this.xc = new Linear(inputSize, hiddenSize, bias);
        this.hc = new Linear(hiddenSize, hiddenSize, bias);

        // Output gate
        this.xo = new Linear(inputSize, hiddenSize, bias);
        this.ho = new Linear(hiddenSize, hiddenSize, bias);

        this.sigmoid = new Sigmoid();
        this.tanh = new Tanh();
    }

    forward(x, h, c) {
        // Forget gate
        const f = this.sigmoid.call(
            this.xf.call(x).add(this.hf.call(h))
        );

        // Input gate
        const i = this.sigmoid.call(
            this.xi.call(x).add(this.hi.call(h))
        );

        // Candidate cell
        const candidate = this.tanh.call(
            this.xc.call(x).add(this.hc.call(h))
        );

        // Updated cell state
        const cNext = f.mul(c).add(
            i.mul(candidate)
        );

        // Output gate
        const o = this.sigmoid.call(
            this.xo.call(x).add(this.ho.call(h))
        );

        // Updated hidden state
        const hNext = o.mul(
            this.tanh.call(cNext)
        );

        return [hNext, cNext];
    }
}