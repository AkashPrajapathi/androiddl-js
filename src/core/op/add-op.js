import { AddBackward } from "../autograd/add-backward.js";
import { Tensor } from "../tensor.js";
import { elementWiseOp, sumToShape } from "../util.js";

export class AddOp {

    apply(a, b) {

        if (!(a instanceof Tensor) || !(b instanceof Tensor)) {
            throw new Error("a and b must be tensors")
        }

        const output = elementWiseOp(a, b, (x, y) => x + y)

        if (a.requiresGrad || b.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = new AddBackward([a, b])
        }
        
        return output
    }
}