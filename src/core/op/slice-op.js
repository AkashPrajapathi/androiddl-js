import { Tensor } from "../tensor.js";
import { normalizeSpec, parseSlice } from "../util.js";
import { SliceBackward } from "../autograd/slice-backward.js";

export class SliceOp {
    apply(a, ...specs) {
        if (!(a instanceof Tensor)) {
            throw new Error("a must be tensor");
        }

        const rank = a.shape.length;

        if (specs.length === 0) {
            specs = Array(rank).fill(":");
        }

        if (specs.length !== rank) {
            throw new Error(
                `Expected ${rank} slice specs, got ${specs.length}`
            );
        }

        const shape = [];
        const strides = [];
        const starts = [];
        const steps = [];
        let offset = a.offset;

        for (let d = 0; d < rank; d++) {
            const { start, stop, step } = parseSlice(specs[d]);
            const norm = normalizeSpec(start, stop, step, a.shape[d]);

            shape.push(norm.length);
            strides.push(a.strides[d] * norm.step);
            starts.push(norm.start);
            steps.push(norm.step);

            offset += norm.start * a.strides[d];
        }

        const output = Tensor.view(a, shape, strides, offset);

        if (a.requiresGrad) {
            output.requiresGrad = true;
            output.gradFn = new SliceBackward(
                [a],
                [...a.shape],
                starts,
                steps
            );
        }

        return output;
    }
}