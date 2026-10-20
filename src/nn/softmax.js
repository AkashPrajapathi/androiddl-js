import { SoftmaxOp } from "../core/op/softmax-op.js"
import { Module } from "./module.js"

export class Softmax extends Module {
    constructor(dim = -1) {
        super()
        this.dim = dim
    }

    forward(input) {
        const op = new SoftmaxOp()
        return op.apply(input, this.dim)
    }
}