import { onesLike } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class TanhBackward extends GradFn {
    constructor(inputs, output) {
        super(inputs)

        this.output = output
    }

    backward(outputGrad) {
        const ones = onesLike(this.output)

        const tanhSqr = this.output.mul(this.output)
        const grad = ones.sub(tanhSqr).mul(outputGrad)

        return [grad]
    }
}