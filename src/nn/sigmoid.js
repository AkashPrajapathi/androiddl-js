import { SigmoidOp } from "../core/op/sigmoid-op.js";
import { Module } from "./module.js";

export class Sigmoid extends Module {
    constructor() {
        super()
        this.op = new SigmoidOp()
    }

    forward(x) {
        return this.op.apply(x)
    }
}