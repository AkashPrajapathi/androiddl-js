import { onesLike } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class SigmoidBackward extends GradFn {
    constructor(inputs, output) {
        super(inputs)

        this.output = output
    }

    backward(outputGrad) {
        const ones = onesLike(this.output.shape)

        const grad = ones.sub(this.output).mul(this.output).mul(outputGrad)

        return [grad]
    }
}