import { GradFn } from "./grad-fn.js";

export class LogBackward extends GradFn {
    constructor(inputs) {
        super(inputs)
    }

    backward(outputGrad) {
        const grad = outputGrad.div(this.inputs[0])
        return [grad]
    }
}