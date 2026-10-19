import { GradFn } from "./grad-fn.js";

export class ExpBackward extends GradFn {
    constructor(inputs, output) {
        super(inputs)
        this.output = output
    }

    backward(outputGrad) {
        const grad = outputGrad.mul(this.output)
        return [grad]
    }
}