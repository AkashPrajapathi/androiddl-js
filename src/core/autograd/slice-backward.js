import { GradFn } from "./grad-fn.js";
import { Tensor } from "../tensor.js";
import { TensorIterator } from "../iterator/tensor-iterator.js";

export class SliceBackward extends GradFn {
    constructor(inputs, inputShape, starts, steps) {
        super(inputs);

        this.inputShape = inputShape;
        this.starts = starts;
        this.steps = steps;
    }

    backward(outputGrad) {
        const input = this.inputs[0];

        const gradInput = new Tensor(
            new Float32Array(
                this.inputShape.reduce((n, dim) => n * dim, 1)
            ),
            [...this.inputShape]
        );

        const iterator = new TensorIterator(
            outputGrad.data,
            outputGrad.shape,
            outputGrad.strides,
            outputGrad.offset
        );

        // Contiguous strides for gradInput's logical shape.
        const inputStrides = new Array(this.inputShape.length);
        let stride = 1;

        for (let d = this.inputShape.length - 1; d >= 0; d--) {
            inputStrides[d] = stride;
            stride *= this.inputShape[d];
        }

        while (iterator.hasNext()) {
            const outputIndex = iterator.getCoordinate();
            const grad = iterator.next();

            let destination = 0;

            for (let d = 0; d < this.inputShape.length; d++) {
                const inputIndex =
                    this.starts[d] +
                    outputIndex[d] * this.steps[d];

                destination += inputIndex * inputStrides[d];
            }

            gradInput.data[destination] += grad;
        }

        return [gradInput];
    }
}