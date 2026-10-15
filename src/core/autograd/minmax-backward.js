import { ReductionIterator } from "../iterator/reduction-iterator.js";
import { Tensor } from "../tensor.js";
import { outputIndex } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class MinMaxBackward extends GradFn {
    constructor(inputs, values, count, axes, keepdim) {
        super(inputs)
        this.values = values
        this.count = count
        this.axes = axes
        this.keepdim = keepdim
    }

    backward(outputGrad) {

        const iter = new ReductionIterator(this.inputs[0].shape, this.inputs[0].strides, this.inputs[0].offset, this.axes, this.keepdim)

        const gradResult = new Float32Array(this.inputs[0].numel)

        while (iter.hasNext()) {

            const pos = iter.next()

            const outIndex = outputIndex(pos.outputCoordinate, outputGrad.shape)
            const inputIndex = outputIndex(pos.inputCoordinate, this.inputs[0].shape)

            const value = this.inputs[0].data[pos.physicalIndex]

            if (value === this.values[outIndex]) {
                gradResult[inputIndex] = outputGrad.data[outIndex] / this.count[outIndex]
            }
        }

        return [new Tensor(gradResult, [...this.inputs[0].shape])]
    }
}