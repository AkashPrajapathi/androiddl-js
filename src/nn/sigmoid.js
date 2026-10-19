import { SigmoidOp } from "../core/op/sigmoid-op.js";
import { Module } from "./module.js";

export class Sigmoid extends Module {
    forward(x) {
        const op = new SigmoidOp()
        return op.apply(x)
    }
}