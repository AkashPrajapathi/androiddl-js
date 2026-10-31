import { UnsqueezeOp } from "./unsqueeze-op.js";
import { ConcatenateOp } from "./concatenate-op.js";

export class StackOp {

    apply(tensors, dim = 0) {

        if (!Array.isArray(tensors) || tensors.length === 0) {
            throw new Error("Expected non-empty tensor list");
        }

        const rank = tensors[0].shape.length;

        if (dim < 0) {
            dim += rank + 1;
        }

        if (dim < 0 || dim > rank) {
            throw new Error(`Invalid dimension: ${dim}`);
        }

        const unsqueeze = new UnsqueezeOp();
        const concatenate = new ConcatenateOp();

        const expanded = tensors.map(
            tensor => unsqueeze.apply(tensor, dim)
        );

        return concatenate.apply(expanded, dim);
    }
}