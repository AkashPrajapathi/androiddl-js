import { Tensor } from "../tensor.js";
import { TensorIterator } from "../iterator/tensor-iterator.js";
import { GradFn } from "./grad-fn.js";

export class SqueezeBackward extends GradFn {
    constructor(inputs) {
        super(inputs);
    }

    backward(gradOutput) {
        const input = this.inputs[0];

        const gradInput = new Tensor(
            new Float32Array(input.numel),
            input.shape
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