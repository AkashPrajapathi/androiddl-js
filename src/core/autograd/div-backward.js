
import { Tensor } from "../tensor.js";
import { sumToShape } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class DivBackward extends GradFn {
    constructor(inputs) {
        super(inputs);
    }

    backward(outputGrad) {
        const [a, b] = this.inputs;

        const gradA = sumToShape(
            outputGrad.div(b),
            a.shape
        );

        const minusOne = new Tensor([-1], []);

        const gradB = sumToShape(
            outputGrad.mul(a).div(b.mul(b)).mul(minusOne),
            b.shape
        );

        return [gradA, gradB];
    }
}
