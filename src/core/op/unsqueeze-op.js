import { UnsqueezeBackward } from "../autograd/unsqueeze-backward.js";
import { Tensor } from "../tensor.js";

export class UnsqueezeOp {
    apply(input, dim) {
        const rank = input.shape.length;

        if (dim < 0) {
            dim += rank + 1;
        }

        if (dim < 0 || dim > rank) {
            throw new Error(`Invalid dimension: ${dim}`);
        }

        const newShape = [...input.shape];
        const newStrides = [...input.strides];

        newShape.splice(dim, 0, 1);

        const newStride = dim < rank
            ? input.strides[dim] * input.shape[dim]
            : 1;

        newStrides.splice(dim, 0, newStride);

        const output = Tensor.view(
            input,
            newShape,
            newStrides,
            input.offset
        );

        if (input.requiresGrad) {
            output.requiresGrad = true;
            output.gradFn = new UnsqueezeBackward([input], dim);
        }

        return output;
    }
}