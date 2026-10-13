import { sumToShape } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class MulBackward extends GradFn {
    constructor(inputs) {
        super(inputs)
    }

    backward(outputGrad) {
        const gradA = sumToShape(outputGrad.mul(this.inputs[1]), this.inputs[0].shape)
        const gradB = sumToShape(outputGrad.mul(this.inputs[0]), this.inputs[1].shape)

        return [gradA, gradB]
    }
}