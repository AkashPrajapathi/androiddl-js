import { Tensor } from "../tensor.js";
import { SqueezeBackward } from "../autograd/squeeze-backward.js";

export class SqueezeOp {
    apply(input, dim = null) {
        if (!(input instanceof Tensor)) {
            throw new Error("Expected a Tensor");
        }

        const rank = input.shape.length;
        const newShape = [];
        const newStrides = [];

        if (dim !== null) {
            if (!Number.isInteger(dim)) {
                throw new Error("Dimension must be an integer");
            }

            if (dim < 0) {
                dim += rank;
            }

            if (dim < 0 || dim >= rank) {
                throw new Error(`Invalid dimension: ${dim}`);
            }
        }

        for (let i = 0; i < rank; i++) {
            const shouldRemove =
                input.shape[i] === 1 &&
                (dim === null || dim === i);

            if (!shouldRemove) {
                newShape.push(input.shape[i]);
                newStrides.push(input.strides[i]);
            }
        }

        const output = Tensor.view(
            input,
            newShape,
            newStrides,
            input.offset
        );

        if (input.requiresGrad) {
            output.requiresGrad = true;
            output.gradFn = new SqueezeBackward([input]);
        }

        return output;
    }
}