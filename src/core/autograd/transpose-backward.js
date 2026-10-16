import { inversePermuatation } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class TransposeBackward extends GradFn {
    constructor(inputs, dim0, dim1) {
        super(inputs)

        this.dim0 = dim0
        this.dim1 = dim1
    }

    backward(outputGrad) {
        const grad = outputGrad.transpose(this.dim0, this.dim1)
        return [grad]
    }
}