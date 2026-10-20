import { SoftmaxOp } from "../core/op/softmax-op.js"
import { Module } from "./module.js"

export class Softmax extends Module {
    constructor(dim = -1) {
        super()
        this.dim = dim
        this.op = new SoftmaxOp()
    }

    forward(x) {
        return this.op.apply(x, this.dim)
    }
}