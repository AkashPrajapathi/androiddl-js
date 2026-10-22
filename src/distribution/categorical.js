import { LogSoftmaxOp } from "../core/op/log-softmax-op.js";
import { SoftmaxOp } from "../core/op/softmax-op.js";
import { Tensor } from "../core/tensor.js";

export class Categorical {
    constructor(logits, dim = -1) {
        if (!(logits instanceof Tensor)) {
            throw new Error("Logits must be a Tensor");
        }

        const rank = logits.shape.length;

        if (rank === 0) {
            throw new Error("Categorical requires non-scalar logits");
        }

        if (!Number.isInteger(dim)) {
            throw new Error("dim must be an integer");
        }

        if (dim < 0) dim += rank;

        if (dim < 0 || dim >= rank) {
            throw new Error("Invalid categorical dimension");
        }

        // Reject invalid logits, including in non-contiguous views.
        for (const value of logits) {
            if (!Number.isFinite(value)) {
                throw new Error("Logits must contain finite values");
            }
        }

        this.logits = logits;
        this.dim = dim;
        this.numCategories = logits.shape[dim];

        this.batchShape = logits.shape.filter(
            (_, i) => i !== dim
        );

        // Reuse existing AndroidDL operations.
        this.probs = new SoftmaxOp().apply(logits, dim);
        this.logProbs = new LogSoftmaxOp().apply(logits, dim);
    }

    sample(random = Math.random) {
        if (typeof random !== "function") {
            throw new Error("random must be a function");
        }

        const shape = this.logits.shape;
        const classes = this.numCategories;

        const outerSize = shape
            .slice(0, this.dim)
            .reduce((a, b) => a * b, 1);

        const innerSize = shape
            .slice(this.dim + 1)
            .reduce((a, b) => a * b, 1);

        const samples = new Array(outerSize * innerSize);
        let sampleIndex = 0;

        for (let outer = 0; outer < outerSize; outer++) {
            for (let inner = 0; inner < innerSize; inner++) {
                const r = random();

                if (!Number.isFinite(r) || r < 0 || r >= 1) {
                    throw new Error(
                        "random must return a value in [0, 1)"
                    );
                }

                let cumulative = 0;
                let lastPositive = -1;
                let selected = -1;

                for (let k = 0; k < classes; k++) {
                    const index =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    const p = this.probs.data[index];

                    if (p > 0) lastPositive = k;

                    cumulative += p;

                    if (r < cumulative && p > 0) {
                        selected = k;
                        break;
                    }
                }

                // Handles floating-point rounding at the tail.
                if (selected === -1) {
                    if (lastPositive === -1) {
                        throw new Error(
                            "Invalid categorical probabilities"
                        );
                    }

                    selected = lastPositive;
                }

                samples[sampleIndex++] = selected;
            }
        }

        return new Tensor(samples, this.batchShape);
    }

    logProb(actions) {
        if (!(actions instanceof Tensor)) {
            throw new Error("Actions must be a Tensor");
        }

        if (
            actions.shape.length !== this.batchShape.length ||
            !actions.shape.every(
                (v, i) => v === this.batchShape[i]
            )
        ) {
            throw new Error(
                `Expected action shape [${this.batchShape}], ` +
                `got [${actions.shape}]`
            );
        }

        const actionValues = Array.from(actions);

        for (const action of actionValues) {
            if (
                !Number.isInteger(action) ||
                action < 0 ||
                action >= this.numCategories
            ) {
                throw new Error(
                    `Actions must be integers in ` +
                    `[0, ${this.numCategories})`
                );
            }
        }

        const shape = this.logits.shape;
        const classes = this.numCategories;

        const outerSize = shape
            .slice(0, this.dim)
            .reduce((a, b) => a * b, 1);

        const innerSize = shape
            .slice(this.dim + 1)
            .reduce((a, b) => a * b, 1);

        // One-hot selection mask. It does not require gradients.
        const mask = new Float32Array(this.logits.numel);
        let sampleIndex = 0;

        for (let outer = 0; outer < outerSize; outer++) {
            for (let inner = 0; inner < innerSize; inner++) {
                const action = actionValues[sampleIndex++];

                const index =
                    outer * classes * innerSize +
                    action * innerSize +
                    inner;

                mask[index] = 1;
            }
        }

        const selection = new Tensor(mask, [...shape]);

        return this.logProbs
            .mul(selection)
            .sum([this.dim]);
    }
}
