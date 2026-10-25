import { inversePermutation } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class PermuteBackward extends GradFn {
    constructor(inputs, dimensions) {
        super(inputs)

        this.dims = dimensions
    }

    backward(outputGrad) {
        const inverse = inversePermutation(this.dims)

        const grad = outputGrad.permute(...inverse)

        return [grad]
    }
}