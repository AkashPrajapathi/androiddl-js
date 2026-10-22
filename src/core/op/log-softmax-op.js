
import { Tensor } from "../tensor.js";
import { LogSoftmaxBackward } from "../autograd/log-softmax-backward.js";

export class LogSoftmaxOp {
    apply(input, dim = -1) {
        if (!(input instanceof Tensor)) {
            throw new Error("Input must be a Tensor");
        }

        const rank = input.shape.length;

        if (rank === 0) {
            throw new Error("LogSoftmax requires a non-scalar tensor");
        }

        if (!Number.isInteger(dim)) {
            throw new Error("dim must be an integer");
        }

        if (dim < 0) dim += rank;

        if (dim < 0 || dim >= rank) {
            throw new Error("Invalid log-softmax dimension");
        }

        const shape = [...input.shape];
        const classes = shape[dim];
        const outputData = new Float32Array(input.numel);

        const outerSize = shape
            .slice(0, dim)
            .reduce((a, b) => a * b, 1);

        const innerSize = shape
            .slice(dim + 1)
            .reduce((a, b) => a * b, 1);

        for (let outer = 0; outer < outerSize; outer++) {
            for (let inner = 0; inner < innerSize; inner++) {
                let max = -Infinity;

                // Find maximum logit.
                for (let k = 0; k < classes; k++) {
                    const index =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    const value = input.data[
                        this.#physicalIndex(input, index)
                    ];

                    if (!Number.isFinite(value)) {
                        throw new Error(
                            "Logits must contain finite values"
                        );
                    }

                    if (value > max) max = value;
                }

                // Stable log-sum-exp.
                let sumExp = 0;

                for (let k = 0; k < classes; k++) {
                    const index =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    const value = input.data[
                        this.#physicalIndex(input, index)
                    ];

                    sumExp += Math.exp(value - max);
                }

                const logNormalizer = max + Math.log(sumExp);

                // Log-softmax output.
                for (let k = 0; k < classes; k++) {
                    const index =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    const value = input.data[
                        this.#physicalIndex(input, index)
                    ];

                    outputData[index] = value - logNormalizer;
                }
            }
        }

        const output = new Tensor(outputData, shape);

        if (input.requiresGrad) {
            output.requiresGrad = true;
            output.gradFn = new LogSoftmaxBackward(
                [input],
                output,
                dim
            );
        }

        return output;
    }

    #physicalIndex(tensor, logicalIndex) {
        let physicalIndex = tensor.offset;

        for (let d = tensor.shape.length - 1; d >= 0; d--) {
            const coordinate = logicalIndex % tensor.shape[d];
            logicalIndex = Math.floor(logicalIndex / tensor.shape[d]);

            physicalIndex += coordinate * tensor.strides[d];
        }

        return physicalIndex;
    }
}
