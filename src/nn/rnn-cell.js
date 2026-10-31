import { Tensor } from "../core/tensor.js";
import { Linear } from "./linear.js";
import { Module } from "./module.js"
import { Tanh } from "./tanh.js";

export class RNNCell extends Module {
    constructor(inputSize, hiddenSize) {
        super();

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.x2h = new Linear(inputSize, hiddenSize);
        this.h2h = new Linear(hiddenSize, hiddenSize);

        this.tanh = new Tanh();
    }

    forward(x, h = null) {
        if (h === null) {
            h = new Tensor(
                new Float32Array(x.shape[0] * this.hiddenSize),
                [x.shape[0], this.hiddenSize]
            );
        }

        return this.tanh.forward(
            this.x2h.forward(x)
                .add(this.h2h.forward(h))
        );
    }
}