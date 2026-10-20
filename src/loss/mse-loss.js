import { Tensor } from "../core/tensor.js";
import { Module } from "../nn/module.js";

export class MSELoss extends Module {
    constructor(reduction = "mean") {
        super();

        if (!["none", "mean", "sum"].includes(reduction)) {
            throw new Error(`Invalid reduction: ${reduction}`);
        }

        this.reduction = reduction;
    }

    forward(prediction, target) {
        if (!(prediction instanceof Tensor) ||
            !(target instanceof Tensor)) {
            throw new Error("Prediction and target must be Tensors");
        }

        if (
            prediction.shape.length !== target.shape.length ||
            !prediction.shape.every(
                (dim, i) => dim === target.shape[i]
            )
        ) {
            throw new Error(
                "Prediction and target must have the same shape"
            );
        }

        const diff = prediction.sub(target);
        const squared = diff.mul(diff);

        if (this.reduction === "none") {
            return squared;
        }

        const axes = prediction.shape.map((_, i) => i);

        if (this.reduction === "sum") {
            return squared.sum(axes);
        }

        return squared.mean(axes);
    }
}
