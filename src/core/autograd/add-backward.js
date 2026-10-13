import { sumToShape } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class AddBackward extends GradFn {
    constructor(inputs) {
        super(inputs)
    }

    backward(outputGrad) {
        const gradA = sumToShape(outputGrad, this.inputs[0].shape)
        const gradB = sumToShape(outputGrad, this.inputs[1].shape)

        return [gradA, gradB]
    }
}