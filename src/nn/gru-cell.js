import { Linear } from "./linear.js";
import { Module } from "./module.js";
import { Sigmoid } from "./sigmoid.js";
import { Tanh } from "./tanh.js";

export class GRUCell extends Module {
    constructor(inputSize, hiddenSize, bias = true) {
        super();

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // Update gate
        this.xz = new Linear(inputSize, hiddenSize, bias);
        this.hz = new Linear(hiddenSize, hiddenSize, bias);

        // Reset gate
        this.xr = new Linear(inputSize, hiddenSize, bias);
        this.hr = new Linear(hiddenSize, hiddenSize, bias);

        // Candidate hidden state
        this.xn = new Linear(inputSize, hiddenSize, bias);
        this.hn = new Linear(hiddenSize, hiddenSize, bias);

        this.sigmoid = new Sigmoid();
        this.tanh = new Tanh();
    }

    forward(x, h) {
        // Update gate
        const z = this.sigmoid.call(
            this.xz.call(x).add(this.hz.call(h))
        );

        // Reset gate
        const r = this.sigmoid.call(
            this.xr.call(x).add(this.hr.call(h))
        );

        // Candidate hidden state: PyTorch-style reset placement
        const n = this.tanh.call(
            this.xn.call(x).add(
                r.mul(this.hn.call(h))
            )
        );

        // h_new = (1 - z) * n + z * h
        const oneMinusZ = z.mul(-1).add(1);

        return oneMinusZ.mul(n).add(z.mul(h));
    }
}