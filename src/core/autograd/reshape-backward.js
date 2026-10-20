import { GradFn } from "./grad-fn.js";

export class ReshapeBackward extends GradFn {
    constructor(inputs) {
        super(inputs)
    }

    backward(outputGrad) {
        const grad = outputGrad.reshape(this.inputs[0].shape)
        return [grad]
    }
}