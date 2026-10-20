import { SoftmaxCrossEntropyOp } from "../core/op/softmax-cross-entropy-op.js";
import { Module } from "../nn/module.js";

export class CrossEntropyLoss extends Module {
    constructor(reduction = "mean", dim = -1) {
        super();

        if (!["none", "mean", "sum"].includes(reduction)) {
            throw new Error(`Invalid reduction: ${reduction}`);
        }

        this.reduction = reduction;
        this.dim = dim;

        this.op = new SoftmaxCrossEntropyOp();
    }

    forward(logits, target) {
        return this.op.apply(
            logits,
            target,
            this.dim,
            this.reduction
        );
    }
}
