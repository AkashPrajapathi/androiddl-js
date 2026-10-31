import { TensorIterator } from "../iterator/tensor-iterator.js";
import { Tensor } from "../tensor.js";
import { GradFn } from "./grad-fn.js";

export class UnsqueezeBackward extends GradFn {
    constructor(inputs, dim) {
        super(inputs);

        this.dim = dim;
    }

    backward(gradOutput) {
        const input = this.inputs[0];

        // Restore original shape
        const gradShape = [...gradOutput.shape];
        const gradStrides = [...gradOutput.strides];

        gradShape.splice(this.dim, 1);
        gradStrides.splice(this.dim, 1);

        // Allocate contiguous gradient
        const gradInput = new Tensor(
            new Float32Array(input.numel),
            gradShape
        );

        const iter = new TensorIterator(
            gradOutput.data,
            gradOutput.shape,
            gradOutput.strides,
            gradOutput.offset
        );

        let index = 0;

        while (iter.hasNext()) {
            gradInput.data[index++] = iter.next();
        }

        return [gradInput];
    }
}