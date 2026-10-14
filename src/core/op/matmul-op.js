import { CoordinateIterator } from "../iterator/coordinate-iterator.js";
import { Tensor } from "../tensor.js";
import { broadcastShape, broadcastStrides, outputIndex } from "../util.js";


export class MatMulOp {


    apply(a, b) {
        const aIsVector = a.shape.length === 1;
        const bIsVector = b.shape.length === 1;

        // Vector × Vector
        if (aIsVector && bIsVector) {
            a = a.reshape([1, a.shape[0]]);
            b = b.reshape([b.shape[0], 1]);

            return this.#applyMatrices(a, b).reshape([]);
        }

        // Vector × Matrix
        if (aIsVector) {
            a = a.reshape([1, a.shape[0]]);
        }

        // Matrix × Vector
        if (bIsVector) {
            b = b.reshape([b.shape[0], 1]);
        }

        const result = this.#applyMatrices(a, b);

        // Remove the temporary vector dimension.
        if (aIsVector) {
            const shape = [...result.shape];
            shape.splice(shape.length - 2, 1);

            return result.reshape(shape);
        }

        if (bIsVector) {
            return result.reshape(result.shape.slice(0, -1));
        }

        return result;
    }

    #applyMatrices(a, b) {
        // Matrix to matrix (rank >= 2)
        if (!(a instanceof Tensor) || !(b instanceof Tensor)) {
            throw new Error("a and b must be tensor");
        }

        const aRank = a.shape.length;
        const bRank = b.shape.length;

        if (aRank < 2 || bRank < 2) {
            throw new Error(
                "a and b must be tensor of rank-2 or more"
            );
        }

        // Extract matrix dimensions
        const m = a.shape[aRank - 2];
        const k = a.shape[aRank - 1];
        const rightK = b.shape[bRank - 2];
        const n = b.shape[bRank - 1];

        if (k !== rightK) {
            throw new Error(
                `Cannot do matmul with [${m}x${k}] and [${rightK}x${n}]`
            );
        }

        // Extract batch shapes
        const aBatchShape = a.shape.slice(0, aRank - 2);
        const bBatchShape = b.shape.slice(0, bRank - 2);

        // Broadcast batch dimensions
        const batchShape = broadcastShape(
            aBatchShape,
            bBatchShape
        );

        // Extract batch strides
        const aBatchStrides = a.strides.slice(0, aRank - 2);
        const bBatchStrides = b.strides.slice(0, bRank - 2);

        // Align strides with broadcasted batch shapes
        const effectiveABatchStrides = broadcastStrides(
            aBatchShape,
            aBatchStrides,
            batchShape
        );

        const effectiveBBatchStrides = broadcastStrides(
            bBatchShape,
            bBatchStrides,
            batchShape
        );

        // Output shape: broadcasted batch + matrix dimensions
        const outputShape = [...batchShape, m, n];

        const result = new Float32Array(
            outputShape.reduce((size, dim) => size * dim, 1)
        );

        // Iterate over broadcasted batches
        const batchIterator = new CoordinateIterator(batchShape);

        while (batchIterator.hasNext()) {
            const batchCoordinate = batchIterator.next();

            let aBatchOffset = a.offset;
            let bBatchOffset = b.offset;

            for (let d = 0; d < batchShape.length; d++) {
                aBatchOffset +=
                    batchCoordinate[d] * effectiveABatchStrides[d];

                bBatchOffset +=
                    batchCoordinate[d] * effectiveBBatchStrides[d];
            }

            const batchIndex = outputIndex(
                batchCoordinate,
                batchShape
            );

            const outputBatchOffset = batchIndex * m * n;

            // Ordinary matrix multiplication for this batch
            for (let i = 0; i < m; i++) {
                for (let j = 0; j < n; j++) {
                    let sum = 0;

                    for (let t = 0; t < k; t++) {
                        const aIndex =
                            aBatchOffset +
                            i * a.strides[aRank - 2] +
                            t * a.strides[aRank - 1];

                        const bIndex =
                            bBatchOffset +
                            t * b.strides[bRank - 2] +
                            j * b.strides[bRank - 1];

                        sum += a.data[aIndex] * b.data[bIndex];
                    }

                    const outIndex =
                        outputBatchOffset + i * n + j;

                    result[outIndex] = sum;
                }
            }
        }

        return new Tensor(result, outputShape);
    }
}
