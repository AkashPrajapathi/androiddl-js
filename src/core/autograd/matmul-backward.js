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

        // const gradA = outputGrad.matmul(this.inputs[1].transpose(rowB, colB))
        // const gradB = this.inputs[0].transpose(rowA, colA).matmul(outputGrad)

        console.log("outputGrad:", outputGrad.toString());

        const bT = this.inputs[1].transpose(rowB, colB);
        console.log("bT shape:", bT.shape);

        const gradA = outputGrad.matmul(bT);
        console.log("raw gradA shape:", gradA.shape);
        console.log("raw gradA:", gradA.toString());

        const aT = this.inputs[0].transpose(rowA, colA);
        console.log("aT shape:", aT.shape);

        const gradB = aT.matmul(outputGrad);
        console.log("raw gradB shape:", gradB.shape);
        console.log("raw gradB:", gradB.toString());

        const reducedA = sumToShape(gradA, shapeA);
        const reducedB = sumToShape(gradB, shapeB);

        console.log("reduced gradA:", reducedA.toString());
        console.log("reduced gradB:", reducedB.toString());

        return [reducedA, reducedB];
    }
}