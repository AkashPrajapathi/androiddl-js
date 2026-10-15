import { sumToShape } from "../util.js";
import { GradFn } from "./grad-fn.js";

export class MatMulBackward extends GradFn {
    constructor(inputs) {
        super(inputs)
    }

    backward(outputGrad) {

        const shapeA = this.inputs[0].shape
        const shapeB = this.inputs[1].shape

        const rowA = shapeA.length - 2
        const colA = shapeA.length - 1

        const rowB = shapeB.length - 2
        const colB = shapeB.length - 1

        const gradA = outputGrad.matmul(this.inputs[1].transpose(rowB, colB))
        const gradB = this.inputs[0].transpose(rowA, colA).matmul(outputGrad)

        const reducedA = sumToShape(gradA, shapeA);
        const reducedB = sumToShape(gradB, shapeB);

        return [reducedA, reducedB];
    }
}