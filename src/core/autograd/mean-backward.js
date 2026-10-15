import { expandReducedAxes } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class MeanBackward extends GradFn {
    constructor(inputs, axes) {
        super(inputs)

        this.axes = axes
    }

    backward(outputGrad) {
        const reshapedShape = [...this.inputs[0].shape]

        let totalElements = 1

        for (const axis of this.axes) {
            reshapedShape[axis] = 1
            totalElements *= this.inputs[0].shape[axis]
        }

        const reshapedGrad = outputGrad.reshape(reshapedShape)

        for (let i = 0; i < outputGrad.numel; i++) {
            reshapedGrad.data[i] /= totalElements
        }

        const grad = expandReducedAxes(reshapedGrad, this.inputs[0].shape)

        return [grad]
    }
}