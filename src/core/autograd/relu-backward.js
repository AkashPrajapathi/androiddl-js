import { Tensor } from "../tensor.js";
import { GradFn } from "./grad-fn.js";

export class ReLUBackward extends GradFn {
    constructor(inputs) {
        super(inputs)
    }

    backward(outputGrad) {
        const result = new Float32Array(outputGrad.numel)

        for (let i = 0; i < outputGrad.numel; i++) {
            result[i] = this.inputs[0].data > 0 ? outputGrad[i] : 0
        }

        const grad = new Tensor(result, [...outputGrad.shape])
        return [grad]
    }
}