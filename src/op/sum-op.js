
import { Tensor } from "../core/tensor.js";
import {
    computeReducedShape,
    outputIndex
} from "../core/util.js";
import { ReductionIterator } from "../iterator/reduction-iterator.js";

export class SumOp {
    apply(tensor, axes, keepdim = false) {
        // Validate tensor.
        if (!(tensor instanceof Tensor)) {
            throw new Error("tensor must be an object of Tensor");
        }

        // Validate axes.
        if (!Array.isArray(axes)) {
            throw new Error("Axes must be a 1D array");
        }

        if (!axes.every(
            axis =>
                Number.isInteger(axis) &&
                axis >= 0 &&
                axis < tensor.shape.length
        )) {
            throw new Error(
                "Axes must contain valid dimension indices"
            );
        }

        if (new Set(axes).size !== axes.length) {
            throw new Error(
                "Axes cannot contain duplicate values"
            );
        }

        // Validate keepdim.
        if (typeof keepdim !== "boolean") {
            throw new Error("keepdim must be a boolean");
        }

        // Calculate output shape.
        const outputShape = computeReducedShape(
            tensor.shape,
            axes,
            keepdim
        );

        const outputSize = outputShape.reduce(
            (acc, dim) => acc * dim,
            1
        );

        if (!Number.isSafeInteger(outputSize)) {
            throw new Error(
                "Output contains too many elements"
            );
        }

        // Allocate zero-initialized output.
        const result = new Float32Array(outputSize);

        // Iterate over input elements and accumulate.
        const iter = new ReductionIterator(
            tensor.shape,
            tensor.strides,
            tensor.offset,
            axes,
            keepdim
        );

        while (iter.hasNext()) {
            const pos = iter.next();

            const value = tensor.data[pos.physicalIndex];

            if (value === undefined) {
                throw new Error(
                    `Invalid physical index: ${pos.physicalIndex}`
                );
            }

            const index = outputIndex(
                outputShape,
                pos.outputCoordinate
            );

            if (
                !Number.isInteger(index) ||
                index < 0 ||
                index >= result.length
            ) {
                throw new Error(
                    `Invalid output index: ${index}`
                );
            }

            result[index] += value;
        }

        return new Tensor(result, outputShape);
    }
}
