
import { Tensor } from "../tensor.js";
import { GradFn } from "./grad-fn.js";

export class LogSoftmaxBackward extends GradFn {
    constructor(inputs, output, dim) {
        super(inputs);

        this.output = output;
        this.shape = [...output.shape];
        this.dim = dim;
    }

    backward(outputGrad) {
        const shape = this.shape;
        const dim = this.dim;
        const classes = shape[dim];

        const outerSize = shape
            .slice(0, dim)
            .reduce((a, b) => a * b, 1);

        const innerSize = shape
            .slice(dim + 1)
            .reduce((a, b) => a * b, 1);

        const gradData = new Float32Array(this.output.numel);

        for (let outer = 0; outer < outerSize; outer++) {
            for (let inner = 0; inner < innerSize; inner++) {
                let gradSum = 0;

                // Sum incoming gradients along the class dimension.
                for (let k = 0; k < classes; k++) {
                    const index =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    gradSum += outputGrad.data[
                        this.#physicalIndex(outputGrad, index)
                    ];
                }

                // dx = dy - exp(logSoftmax(x)) * sum(dy)
                for (let k = 0; k < classes; k++) {
                    const index =
                        outer * classes * innerSize +
                        k * innerSize +
                        inner;

                    const logProbability = this.output.data[index];
                    const probability = Math.exp(logProbability);

                    const upstream = outputGrad.data[
                        this.#physicalIndex(outputGrad, index)
                    ];

                    gradData[index] =
                        upstream - probability * gradSum;
                }
            }
        }

        return [new Tensor(gradData, shape)];
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
