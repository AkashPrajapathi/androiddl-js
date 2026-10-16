import { Tensor } from "../core/tensor.js"

export class Parameter extends Tensor {
    constructor(data, shape) {
        super(data, shape, true)
    }
}