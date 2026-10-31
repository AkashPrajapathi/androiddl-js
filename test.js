import { Tensor, RNNCell, RNN, Linear, GRUCell, GRU, LSTMCell, LSTM } from "./androiddl.js";

let passed = 0;
let failed = 0;

function log(message) {
    console.log(message);
}

function test(name, fn) {
    try {
        fn();
        passed++;
        log(`✅ PASS: ${name}`);
    } catch (error) {
        failed++;
        log(`❌ FAIL: ${name}\n   ${error.message}`);
    }
}

function makeTensor(data, shape, requiresGrad = false) {
    const x = new Tensor(Float32Array.from(data), shape);
    x.requiresGrad = requiresGrad
    return x
}

function values(tensor) {
    return Array.from(tensor);
}

function assert(condition, message) {
    if (!condition) throw new Error(message);
}

function assertArray(actual, expected, message) {
    assert(
        JSON.stringify(Array.from(actual)) === JSON.stringify(expected),
        `${message}: got ${JSON.stringify(Array.from(actual))}, expected ${JSON.stringify(expected)}`
    );
}

function assertTensor(tensor, shape, expected, message) {
    assertArray(tensor.shape, shape, `${message} shape`);
    assertArray(values(tensor), expected, `${message} values`);
}

function assertThrows(fn, message) {
    let threw = false;

    try {
        fn();
    } catch {
        threw = true;
    }

    assert(threw, `${message}: expected an exception`);
}


// ==================================================
// 1. BASIC CONCATENATION
// ==================================================

test("1D concatenation", () => {
    const a = makeTensor([1, 2, 3], [3]);
    const b = makeTensor([4, 5], [2]);

    assertTensor(
        Tensor.cat([a, b], 0),
        [5],
        [1, 2, 3, 4, 5],
        "1D concatenation"
    );
});

test("2D concatenation along dim 0", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6], [1, 2]);

    assertTensor(
        Tensor.cat([a, b], 0),
        [3, 2],
        [1, 2, 3, 4, 5, 6],
        "2D dim 0"
    );
});

test("2D concatenation along dim 1", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6, 7, 8], [2, 2]);

    assertTensor(
        Tensor.cat([a, b], 1),
        [2, 4],
        [1, 2, 5, 6, 3, 4, 7, 8],
        "2D dim 1"
    );
});

test("Multiple tensors concatenation", () => {
    const a = makeTensor([1, 2], [2]);
    const b = makeTensor([3, 4], [2]);
    const c = makeTensor([5, 6], [2]);

    assertTensor(
        Tensor.cat([a, b, c], 0),
        [6],
        [1, 2, 3, 4, 5, 6],
        "Multiple tensors"
    );
});


// ==================================================
// 2. HIGHER DIMENSIONAL CONCATENATION
// ==================================================

test("3D concatenation along dim 0", () => {
    const a = makeTensor([1, 2, 3, 4], [1, 2, 2]);
    const b = makeTensor([5, 6, 7, 8], [1, 2, 2]);

    assertTensor(
        Tensor.cat([a, b], 0),
        [2, 2, 2],
        [1, 2, 3, 4, 5, 6, 7, 8],
        "3D dim 0"
    );
});

test("3D concatenation along dim 1", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 1, 2]);
    const b = makeTensor([5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 2]);

    assertTensor(
        Tensor.cat([a, b], 1),
        [2, 3, 2],
        [1, 2, 5, 6, 7, 8, 3, 4, 9, 10, 11, 12],
        "3D dim 1"
    );
});

test("3D concatenation along dim 2", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 1, 2]);
    const b = makeTensor([5, 6, 7, 8], [2, 1, 2]);

    assertTensor(
        Tensor.cat([a, b], 2),
        [2, 1, 4],
        [1, 2, 5, 6, 3, 4, 7, 8],
        "3D dim 2"
    );
});


// ==================================================
// 3. NEGATIVE DIMENSIONS
// ==================================================

test("Negative dimension -1", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6], [2, 1]);

    assertTensor(
        Tensor.cat([a, b], -1),
        [2, 3],
        [1, 2, 5, 3, 4, 6],
        "Negative dim -1"
    );
});

test("Negative dimension -2", () => {
    const a = makeTensor([1, 2], [1, 2]);
    const b = makeTensor([3, 4, 5, 6], [2, 2]);

    assertTensor(
        Tensor.cat([a, b], -2),
        [3, 2],
        [1, 2, 3, 4, 5, 6],
        "Negative dim -2"
    );
});


// ==================================================
// 4. NON-CONTIGUOUS INPUTS
// ==================================================

test("Concatenate sliced non-contiguous tensor", () => {
    const base = makeTensor(
        [0, 1, 2, 3, 4, 5, 6, 7],
        [2, 4]
    );

    const a = base.slice(":", "::2");
    const b = makeTensor([100, 200], [2, 1]);

    assertTensor(
        Tensor.cat([a, b], 1),
        [2, 3],
        [0, 2, 100, 4, 6, 200],
        "Non-contiguous slice"
    );
});

test("Concatenate reversed tensor", () => {
    const base = makeTensor([1, 2, 3, 4, 5, 6], [2, 3]);

    const a = base.slice(":", "::-1");
    const b = makeTensor([7, 8, 9, 10, 11, 12], [2, 3]);

    assertTensor(
        Tensor.cat([a, b], 1),
        [2, 6],
        [3, 2, 1, 7, 8, 9, 6, 5, 4, 10, 11, 12],
        "Reversed input"
    );
});

test("Concatenate permuted tensor", () => {
    const base = makeTensor([1, 2, 3, 4, 5, 6], [2, 3]);

    const a = base.permute(1, 0);
    const b = makeTensor([7, 8, 9, 10, 11, 12], [3, 2]);

    assertTensor(
        Tensor.cat([a, b], 1),
        [3, 4],
        [1, 4, 7, 8, 2, 5, 9, 10, 3, 6, 11, 12],
        "Permuted input"
    );
});


// ==================================================
// 5. ZERO-SIZED TENSORS
// ==================================================

test("Zero-sized tensor before non-empty tensor", () => {
    const a = makeTensor([], [2, 0]);
    const b = makeTensor([1, 2, 3, 4], [2, 2]);

    assertTensor(
        Tensor.cat([a, b], 1),
        [2, 2],
        [1, 2, 3, 4],
        "Empty tensor first"
    );
});

test("Zero-sized tensor between non-empty tensors", () => {
    const a = makeTensor([1, 2], [2, 1]);
    const b = makeTensor([], [2, 0]);
    const c = makeTensor([3, 4], [2, 1]);

    assertTensor(
        Tensor.cat([a, b, c], 1),
        [2, 2],
        [1, 3, 2, 4],
        "Empty tensor in middle"
    );
});

test("All tensors have zero elements", () => {
    const a = makeTensor([], [2, 0]);
    const b = makeTensor([], [2, 0]);

    assertTensor(
        Tensor.cat([a, b], 1),
        [2, 0],
        [],
        "All empty tensors"
    );
});

test("Zero-sized non-concatenation dimension", () => {
    const a = makeTensor([], [0, 2]);
    const b = makeTensor([], [0, 3]);

    assertTensor(
        Tensor.cat([a, b], 1),
        [0, 5],
        [],
        "Zero non-cat dimension"
    );
});


// ==================================================
// 6. SINGLE INPUT AND STORAGE
// ==================================================

test("Single tensor concatenation", () => {
    const a = makeTensor([1, 2, 3], [3]);

    const result = Tensor.cat([a], 0);

    assertTensor(result, [3], [1, 2, 3], "Single tensor");
});

test("Output has independent storage", () => {
    const a = makeTensor([1, 2, 3], [3]);

    const result = Tensor.cat([a], 0);

    assert(
        result.data !== a.data,
        "Output shares storage with input"
    );

    result.data[0] = 999;

    assert(
        a.data[0] === 1,
        "Modifying output modified input"
    );
});

test("Output is contiguous", () => {
    const base = makeTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const a = base.permute(1, 0);

    const result = Tensor.cat([a], 0);

    assertArray(
        result.strides,
        [2, 1],
        "Output contiguous strides"
    );
});


// ==================================================
// 7. VALIDATION TESTS
// ==================================================

test("Empty tensor list throws", () => {
    assertThrows(
        () => Tensor.cat([], 0),
        "Empty tensor list"
    );
});

test("Non-tensor input throws", () => {
    const a = makeTensor([1, 2], [2]);

    assertThrows(
        () => Tensor.cat([a, [3, 4]], 0),
        "Non-tensor input"
    );
});

test("Different ranks throw", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6], [2]);

    assertThrows(
        () => Tensor.cat([a, b], 0),
        "Different ranks"
    );
});

test("Invalid positive dimension throws", () => {
    const a = makeTensor([1, 2], [2]);

    assertThrows(
        () => Tensor.cat([a], 1),
        "Invalid positive dimension"
    );
});

test("Invalid negative dimension throws", () => {
    const a = makeTensor([1, 2], [2]);

    assertThrows(
        () => Tensor.cat([a], -2),
        "Invalid negative dimension"
    );
});

test("Mismatched non-cat dimension throws", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6, 7], [1, 3]);

    assertThrows(
        () => Tensor.cat([a, b], 0),
        "Mismatched non-cat dimension"
    );
});

// ==================================================
// 8. CONCATENATION BACKWARD TESTS
// ==================================================

test("1D concatenation backward", () => {
    const a = makeTensor([1, 2, 3], [3]);
    const b = makeTensor([4, 5], [2]);

    a.requiresGrad = true;
    b.requiresGrad = true;

    const output = Tensor.cat([a, b], 0);

    const weights = makeTensor([10, 20, 30, 40, 50], [5]);

    const loss = output.mul(weights).sum();
    loss.backward();

    assertTensor(
        a.grad,
        [3],
        [10, 20, 30],
        "Gradient of input a"
    );

    assertTensor(
        b.grad,
        [2],
        [40, 50],
        "Gradient of input b"
    );
});

test("2D concatenation backward along dim 0", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6], [1, 2]);

    a.requiresGrad = true;
    b.requiresGrad = true;

    const output = Tensor.cat([a, b], 0);

    const weights = makeTensor(
        [10, 20, 30, 40, 50, 60],
        [3, 2]
    );

    const loss = output.mul(weights).sum();
    loss.backward();

    assertTensor(
        a.grad,
        [2, 2],
        [10, 20, 30, 40],
        "Gradient of input a"
    );

    assertTensor(
        b.grad,
        [1, 2],
        [50, 60],
        "Gradient of input b"
    );
});

test("2D concatenation backward along dim 1", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6, 7, 8], [2, 2]);

    a.requiresGrad = true;
    b.requiresGrad = true;

    const output = Tensor.cat([a, b], 1);

    const weights = makeTensor(
        [10, 20, 30, 40, 50, 60, 70, 80],
        [2, 4]
    );

    const loss = output.mul(weights).sum();
    loss.backward();

    assertTensor(
        a.grad,
        [2, 2],
        [10, 20, 50, 60],
        "Gradient of input a"
    );

    assertTensor(
        b.grad,
        [2, 2],
        [30, 40, 70, 80],
        "Gradient of input b"
    );
});

test("Multiple tensors concatenation backward", () => {
    const a = makeTensor([1, 2], [2]);
    const b = makeTensor([3, 4], [2]);
    const c = makeTensor([5, 6], [2]);

    a.requiresGrad = true;
    b.requiresGrad = true;
    c.requiresGrad = true;

    const output = Tensor.cat([a, b, c], 0);

    const weights = makeTensor(
        [10, 20, 30, 40, 50, 60],
        [6]
    );

    const loss = output.mul(weights).sum();
    loss.backward();

    assertTensor(a.grad, [2], [10, 20], "Gradient a");
    assertTensor(b.grad, [2], [30, 40], "Gradient b");
    assertTensor(c.grad, [2], [50, 60], "Gradient c");
});

test("Backward with negative dimension", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6], [2, 1]);

    a.requiresGrad = true;
    b.requiresGrad = true;

    const output = Tensor.cat([a, b], -1);

    const weights = makeTensor(
        [10, 20, 30, 40, 50, 60],
        [2, 3]
    );

    const loss = output.mul(weights).sum();
    loss.backward();

    assertTensor(
        a.grad,
        [2, 2],
        [10, 20, 40, 50],
        "Gradient of input a"
    );

    assertTensor(
        b.grad,
        [2, 1],
        [30, 60],
        "Gradient of input b"
    );
});

test("Backward with non-contiguous input", () => {
    const base = makeTensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]
    );

    const a = base.permute(1, 0);
    const b = makeTensor([7, 8, 9], [3, 1]);

    a.requiresGrad = true;
    b.requiresGrad = true;

    const output = Tensor.cat([a, b], 1);

    const weights = makeTensor(
        [10, 20, 30,
            40, 50, 60,
            70, 80, 90],
        [3, 3]
    );

    const loss = output.mul(weights).sum();
    loss.backward();

    assertTensor(
        a.grad,
        [3, 2],
        [10, 20, 40, 50, 70, 80],
        "Gradient of permuted input"
    );

    assertTensor(
        b.grad,
        [3, 1],
        [30, 60, 90],
        "Gradient of input b"
    );
});

test("Backward with zero-sized input", () => {
    const a = makeTensor([], [2, 0]);
    const b = makeTensor([1, 2, 3, 4], [2, 2]);

    a.requiresGrad = true;
    b.requiresGrad = true;

    const output = Tensor.cat([a, b], 1);

    const weights = makeTensor(
        [10, 20, 30, 40],
        [2, 2]
    );

    const loss = output.mul(weights).sum();
    loss.backward();

    assertTensor(
        a.grad,
        [2, 0],
        [],
        "Gradient of empty input"
    );

    assertTensor(
        b.grad,
        [2, 2],
        [10, 20, 30, 40],
        "Gradient of non-empty input"
    );
});


function assertEqual(actual, expected, message) {
    assert(
        actual === expected,
        `${message}: got ${actual}, expected ${expected}`
    );
}

// ==================================================
// UNSQUEEZE FORWARD TESTS
// ==================================================

test("Unsqueeze at dim 0", () => {
    const x = makeTensor([1, 2, 3], [3]);

    const y = x.unsqueeze(0);

    assertTensor(y, [1, 3], [1, 2, 3], "Unsqueeze dim 0");
});

test("Unsqueeze at dim 1", () => {
    const x = makeTensor([1, 2, 3], [3]);

    const y = x.unsqueeze(1);

    assertTensor(y, [3, 1], [1, 2, 3], "Unsqueeze dim 1");
});

test("Unsqueeze with negative dim", () => {
    const x = makeTensor([1, 2, 3], [3]);

    const y = x.unsqueeze(-1);

    assertTensor(y, [3, 1], [1, 2, 3], "Unsqueeze -1");
});

test("Unsqueeze 3D tensor", () => {
    const x = makeTensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]
    );

    const y = x.unsqueeze(1);

    assertTensor(
        y,
        [2, 1, 3],
        [1, 2, 3, 4, 5, 6],
        "Unsqueeze 3D"
    );
});

test("Unsqueeze permuted tensor", () => {
    const base = makeTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const x = base.permute(1, 0);

    const y = x.unsqueeze(1);
    assertTensor(
        y,
        [3, 1, 2],
        [1, 4, 2, 5, 3, 6],
        "Unsqueeze permuted"
    );
});

test("Unsqueeze reversed tensor", () => {
    const base = makeTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const x = base.slice(":", "::-1");

    const y = x.unsqueeze(0);

    assertTensor(
        y,
        [1, 2, 3],
        [3, 2, 1, 6, 5, 4],
        "Unsqueeze reversed"
    );

    assert(
        y.strides[1] === x.strides[0],
        "Original negative stride was not preserved"
    );
});

test("Unsqueeze shares storage", () => {
    const x = makeTensor([1, 2, 3], [3]);
    const y = x.unsqueeze(0);

    assert(y.data === x.data, "Storage was copied");

    y.data[0] = 100;

    assertEqual(x.data[0], 100, "Shared storage");
});

test("Invalid unsqueeze dimension", () => {
    const x = makeTensor([1, 2, 3], [3]);

    assertThrows(() => x.unsqueeze(2), "Invalid positive dim");
    assertThrows(() => x.unsqueeze(-3), "Invalid negative dim");
});


// ==================================================
// UNSQUEEZE BACKWARD TESTS
// ==================================================

test("Unsqueeze backward dim 0", () => {
    const x = makeTensor([1, 2, 3], [3]);
    x.requiresGrad = true;

    const y = x.unsqueeze(0);
    const loss = y.mul(
        makeTensor([10, 20, 30], [1, 3])
    ).sum();

    loss.backward();

    assertTensor(
        x.grad,
        [3],
        [10, 20, 30],
        "Unsqueeze backward dim 0"
    );
});

test("Unsqueeze backward dim 1", () => {
    const x = makeTensor([1, 2, 3], [3]);
    x.requiresGrad = true;

    const y = x.unsqueeze(1);
    const loss = y.mul(
        makeTensor([10, 20, 30], [3, 1])
    ).sum();

    loss.backward();

    assertTensor(
        x.grad,
        [3],
        [10, 20, 30],
        "Unsqueeze backward dim 1"
    );
});

test("Unsqueeze backward with permuted input", () => {
    const base = makeTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const x = base.permute(1, 0);
    x.requiresGrad = true;

    const y = x.unsqueeze(1);

    const weights = makeTensor(
        [10, 20, 30, 40, 50, 60],
        [3, 1, 2]
    );

    y.mul(weights).sum().backward();

    assertTensor(
        x.grad,
        [3, 2],
        [10, 20, 30, 40, 50, 60],
        "Permuted backward"
    );
});

test("Unsqueeze backward with reversed input", () => {
    const base = makeTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const x = base.slice(":", "::-1");
    x.requiresGrad = true;

    const y = x.unsqueeze(0);

    const weights = makeTensor(
        [10, 20, 30, 40, 50, 60],
        [1, 2, 3]
    );

    y.mul(weights).sum().backward();

    assertTensor(
        x.grad,
        [2, 3],
        [10, 20, 30, 40, 50, 60],
        "Reversed backward"
    );
});

test("Stack 1D tensors dim 0", () => {
    const a = makeTensor([1, 2, 3], [3]);
    const b = makeTensor([4, 5, 6], [3]);

    const result = Tensor.stack([a, b], 0);

    assertTensor(result, [2, 3], [1, 2, 3, 4, 5, 6]);
});

test("Stack 1D tensors dim 1", () => {
    const a = makeTensor([1, 2, 3], [3]);
    const b = makeTensor([4, 5, 6], [3]);

    const result = Tensor.stack([a, b], 1);

    assertTensor(result, [3, 2], [1, 4, 2, 5, 3, 6]);
});

test("Stack 2D tensors dim 0", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6, 7, 8], [2, 2]);

    const result = Tensor.stack([a, b], 0);

    assertTensor(result, [2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
});

test("Stack 2D tensors dim 1", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6, 7, 8], [2, 2]);

    const result = Tensor.stack([a, b], 1);

    assertTensor(result, [2, 2, 2], [1, 2, 5, 6, 3, 4, 7, 8]);
});

test("Stack 2D tensors dim 2", () => {
    const a = makeTensor([1, 2, 3, 4], [2, 2]);
    const b = makeTensor([5, 6, 7, 8], [2, 2]);

    const result = Tensor.stack([a, b], 2);

    assertTensor(result, [2, 2, 2], [1, 5, 2, 6, 3, 7, 4, 8]);
});

test("Stack with negative dim", () => {
    const a = makeTensor([1, 2, 3], [3]);
    const b = makeTensor([4, 5, 6], [3]);

    const result = Tensor.stack([a, b], -1);

    assertTensor(result, [3, 2], [1, 4, 2, 5, 3, 6]);
});

test("Stack multiple tensors", () => {
    const a = makeTensor([1, 2], [2]);
    const b = makeTensor([3, 4], [2]);
    const c = makeTensor([5, 6], [2]);

    const result = Tensor.stack([a, b, c], 0);

    assertTensor(result, [3, 2], [1, 2, 3, 4, 5, 6]);
});

test("Stack non-contiguous tensors", () => {
    const a = makeTensor([1, 2, 3, 4, 5, 6], [2, 3]).permute(1, 0);
    const b = makeTensor([7, 8, 9, 10, 11, 12], [2, 3]).permute(1, 0);

    const result = Tensor.stack([a, b], 0);

    assertTensor(result, [2, 3, 2], [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12]);
});

test("Stack rejects empty list", () => {
    assertThrows(() => Tensor.stack([]));
});

test("Stack rejects mismatched shapes", () => {
    const a = makeTensor([1, 2], [2]);
    const b = makeTensor([1, 2, 3], [3]);

    assertThrows(() => Tensor.stack([a, b]));
});

test("Stack rejects invalid dimension", () => {
    const a = makeTensor([1, 2], [2]);
    const b = makeTensor([3, 4], [2]);

    assertThrows(() => Tensor.stack([a, b], 2));
});

test("Stack backward dim 0", () => {
    const a = makeTensor([1, 2, 3], [3], true);
    const b = makeTensor([4, 5, 6], [3], true);

    const result = Tensor.stack([a, b], 0);

    result.backward(
        makeTensor([1, 2, 3, 4, 5, 6], [2, 3])
    );

    assertArray(a.grad.data, [1, 2, 3]);
    assertArray(b.grad.data, [4, 5, 6]);
});

test("Stack backward dim 1", () => {
    const a = makeTensor([1, 2, 3], [3], true);
    const b = makeTensor([4, 5, 6], [3], true);

    const result = Tensor.stack([a, b], 1);

    result.backward(
        makeTensor([1, 2, 3, 4, 5, 6], [3, 2])
    );

    assertArray(a.grad.data, [1, 3, 5]);
    assertArray(b.grad.data, [2, 4, 6]);
});

test("Stack backward multiple tensors", () => {
    const a = makeTensor([1, 2], [2], true);
    const b = makeTensor([3, 4], [2], true);
    const c = makeTensor([5, 6], [2], true);

    const result = Tensor.stack([a, b, c], 0);

    result.backward(
        makeTensor([1, 2, 3, 4, 5, 6], [3, 2])
    );

    assertArray(a.grad.data, [1, 2]);
    assertArray(b.grad.data, [3, 4]);
    assertArray(c.grad.data, [5, 6]);
});

test("Stack backward negative dim", () => {
    const a = makeTensor([1, 2, 3], [3], true);
    const b = makeTensor([4, 5, 6], [3], true);

    const result = Tensor.stack([a, b], -1);

    result.backward(
        makeTensor([1, 2, 3, 4, 5, 6], [3, 2])
    );

    assertArray(a.grad.data, [1, 3, 5]);
    assertArray(b.grad.data, [2, 4, 6]);
});

test("Stack backward through permuted inputs", () => {
    const baseA = makeTensor([1, 2, 3, 4, 5, 6], [2, 3], true);
    const baseB = makeTensor([7, 8, 9, 10, 11, 12], [2, 3], true);

    const a = baseA.permute(1, 0);
    const b = baseB.permute(1, 0);

    const result = Tensor.stack([a, b], 0);

    result.backward(
        makeTensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [2, 3, 2]
        )
    );

    assertArray(baseA.grad.data, [1, 2, 3, 4, 5, 6]);
    assertArray(baseB.grad.data, [7, 8, 9, 10, 11, 12]);
});

// =================================================
// Squeeze
// =================================================
test("Squeeze all singleton dimensions", () => {
    const x = makeTensor(
        [1, 2, 3, 4, 5, 6],
        [1, 2, 1, 3]
    );

    const y = x.squeeze();

    assertTensor(y, [2, 3], [1, 2, 3, 4, 5, 6]);
});

test("Squeeze dimension 0", () => {
    const x = makeTensor([1, 2, 3], [1, 3]);

    assertTensor(x.squeeze(0), [3], [1, 2, 3]);
});

test("Squeeze negative dimension", () => {
    const x = makeTensor([1, 2, 3], [3, 1]);

    assertTensor(x.squeeze(-1), [3], [1, 2, 3]);
});

test("Squeeze non-singleton dimension is no-op", () => {
    const x = makeTensor([1, 2, 3, 4], [2, 2]);

    assertTensor(x.squeeze(0), [2, 2], [1, 2, 3, 4]);
});

test("Squeeze preserves storage", () => {
    const x = makeTensor([1, 2, 3], [1, 3]);
    const y = x.squeeze();

    if (x.data !== y.data) {
        throw new Error("Storage is not shared");
    }
});

test("Squeeze non-contiguous tensor", () => {
    const x = makeTensor(
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3]
    ).permute(0, 2, 1);

    const y = x.squeeze(0);

    assertTensor(y, [3, 2], [1, 4, 2, 5, 3, 6]);
});

// Backward
test("Squeeze backward all singleton dimensions", () => {
    const x = makeTensor([1, 2], [1, 2, 1], true);

    const y = x.squeeze();

    y.backward(makeTensor([10, 20], [2]));

    assertArray(x.grad.data, [10, 20]);
    assertArray(x.grad.shape, [1, 2, 1]);
});

test("Squeeze backward dimension 0", () => {
    const x = makeTensor([1, 2, 3], [1, 3], true);

    x.squeeze(0).backward(
        makeTensor([4, 5, 6], [3])
    );

    assertArray(x.grad.data, [4, 5, 6]);
});

test("Squeeze backward negative dimension", () => {
    const x = makeTensor([1, 2, 3], [3, 1], true);

    x.squeeze(-1).backward(
        makeTensor([4, 5, 6], [3])
    );

    assertArray(x.grad.data, [4, 5, 6]);
});

test("Squeeze backward through permuted input", () => {
    const base = makeTensor(
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3],
        true
    );

    const x = base.permute(0, 2, 1);
    const y = x.squeeze(0);

    y.backward(
        makeTensor([1, 2, 3, 4, 5, 6], [3, 2])
    );

    assertArray(
        base.grad.data,
        [1, 2, 3, 4, 5, 6]
    );
});

// =================================================
// RNN
// =================================================
test("RNNCell forward with automatic hidden state", () => {
    const cell = new RNNCell(3, 4);

    const x = makeTensor([
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6
    ], [2, 3]);

    const output = cell.call(x);

    assertArray(output.shape, [2, 4]);
});

test("RNNCell forward with explicit hidden state", () => {
    const cell = new RNNCell(3, 4);

    const x = makeTensor([
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6
    ], [2, 3]);

    const h = makeTensor([
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8
    ], [2, 4]);

    const output = cell.call(x, h);

    assertArray(output.shape, [2, 4]);
});

test("RNNCell backward", () => {
    const cell = new RNNCell(3, 4);

    const x = makeTensor([
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6
    ], [2, 3], true);

    const h = makeTensor([
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8
    ], [2, 4], true);

    const output = cell.call(x, h);

    output.backward(
        makeTensor(new Array(8).fill(1), [2, 4])
    );

    if (x.grad === null) {
        throw new Error("Input gradient missing");
    }

    if (h.grad === null) {
        throw new Error("Hidden gradient missing");
    }

    if (cell.x2h.w.grad === null) {
        throw new Error("Input weight gradient missing");
    }

    if (cell.h2h.w.grad === null) {
        throw new Error("Hidden weight gradient missing");
    }
});

test("RNN backward through multiple timesteps", () => {
    const rnn = new RNN(3, 4);

    const input = makeTensor(
        Array.from({ length: 18 }, (_, i) => (i + 1) * 0.01),
        [3, 2, 3],
        true
    );

    const h0 = makeTensor(
        Array.from({ length: 8 }, (_, i) => (i + 1) * 0.01),
        [2, 4],
        true
    );

    const [output, hn] = rnn.forward(input, h0);

    output.backward(
        makeTensor(new Array(24).fill(1), [3, 2, 4])
    );
    assertArray(output.shape, [3, 2, 4]);
    assertArray(hn.shape, [2, 4]);

    if (input.grad === null) {
        throw new Error("Input gradient missing");
    }

    if (h0.grad === null) {
        throw new Error("Initial hidden gradient missing");
    }

    if (rnn.cell.x2h.w.grad === null) {
        throw new Error("Input weight gradient missing");
    }

    if (rnn.cell.h2h.w.grad === null) {
        throw new Error("Hidden weight gradient missing");
    }
});

test("RNN non-contiguous input", () => {
    const rnn = new RNN(3, 4);

    // Create contiguous base tensor
    const base = makeTensor(
        Array.from({ length: 18 }, (_, i) => (i + 1) * 0.01),
        [3, 3, 2],
        true
    );

    // Permute to [seq, batch, input]
    const input = base.permute(0, 2, 1);

    assertArray(input.shape, [3, 2, 3]);
    assertArray(input.strides, [6, 1, 2]);

    // Forward
    const [output, hn] = rnn.forward(input);

    assertArray(output.shape, [3, 2, 4]);
    assertArray(hn.shape, [2, 4]);

    // Backward
    const grad = makeTensor(
        new Array(24).fill(1),
        [3, 2, 4]
    );

    output.backward(grad);

    // Gradients must flow through permute into original storage
    assertArray(base.grad.shape, [3, 3, 2]);

    const finite = base.grad.data.every(Number.isFinite);
    assert(finite, "Input gradients must be finite");

    const nonZero = base.grad.data.some(v => Math.abs(v) > 1e-8);
    assert(nonZero, "Input gradients must not all be zero");
})


test("RNN initial hidden state backward", () => {
    const rnn = new RNN(3, 4);

    const input = makeTensor(
        Array.from({ length: 18 }, (_, i) => (i + 1) * 0.01),
        [3, 2, 3],
        true
    );

    const h0 = makeTensor(
        Array.from({ length: 8 }, (_, i) => (i + 1) * 0.01),
        [2, 4],
        true
    );

    // Forward
    const [output, hn] = rnn.forward(input, h0);

    assertArray(output.shape, [3, 2, 4]);
    assertArray(hn.shape, [2, 4]);

    // Backward
    const grad = makeTensor(
        new Array(24).fill(1),
        [3, 2, 4]
    );

    output.backward(grad);

    // Initial hidden state must receive gradients
    assertArray(h0.grad.shape, [2, 4]);

    const finite = h0.grad.data.every(Number.isFinite);
    assert(finite, "h0 gradients must be finite");

    const nonZero = h0.grad.data.some(v => Math.abs(v) > 1e-8);
    assert(nonZero, "h0 gradients must not all be zero");
});

test("RNN with empty sequence", () => {
    const rnn = new RNN(3, 4);

    const input = makeTensor(
        [],
        [0, 2, 3],
        true
    );

    const [output, hn] = rnn.forward(input);

    assertArray(output.shape, [0, 2, 4]);
    assertArray(hn.shape, [2, 4]);

    assert(
        hn.data.every(v => v === 0),
        "Final hidden state must be zero"
    );
});

test("RNN with zero-sized batch", () => {
    const rnn = new RNN(3, 4);

    const input = makeTensor(
        [],
        [3, 0, 3],
        true
    );

    const [output, hn] = rnn.forward(input);

    assertArray(output.shape, [3, 0, 4]);
    assertArray(hn.shape, [0, 4]);

    assert(
        output.numel === 0,
        "Output must contain zero elements"
    );

    assert(
        hn.numel === 0,
        "Hidden state must contain zero elements"
    );
});

test("Linear with zero-sized batch", () => {
    const linear = new Linear(3, 4);

    const input = makeTensor([], [0, 3], true);

    const output = linear.forward(input);

    assertArray(output.shape, [0, 4]);
});

test("MatMul with zero-sized batch", () => {
    const a = makeTensor([], [0, 3], true);

    const b = makeTensor(
        Array.from({ length: 12 }, (_, i) => (i + 1) * 0.01),
        [3, 4],
        true
    );

    const output = a.matmul(b);

    assertArray(output.shape, [0, 4]);
    assert(output.numel === 0, "Output must be empty");
});

test("RNN batch independence", () => {
    const rnn = new RNN(3, 4);

    const batchSize = 2;
    const seqLength = 3;

    const input = makeTensor(
        [
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,

            0.7, 0.8, 0.9,
            1.0, 1.1, 1.2,

            1.3, 1.4, 1.5,
            1.6, 1.7, 1.8
        ],
        [seqLength, batchSize, 3],
        false
    );

    const [batchOutput] = rnn.forward(input);

    for (let b = 0; b < batchSize; b++) {
        const singleInput = input.slice(
            ":",
            `${b}:${b + 1}`,
            ":"
        );

        const [singleOutput] = rnn.forward(singleInput);

        for (let t = 0; t < seqLength; t++) {
            for (let h = 0; h < 4; h++) {
                const batchIndex = (t * batchSize + b) * 4 + h;
                const singleIndex = t * 4 + h;

                assert(
                    Math.abs(
                        batchOutput.data[batchIndex] -
                        singleOutput.data[singleIndex]
                    ) < 1e-5,
                    `Mismatch at timestep ${t}, batch ${b}, hidden ${h}`
                );
            }
        }
    }
});

test("RNN resets hidden state between forward calls", () => {
    const rnn = new RNN(3, 4);

    const inputA = makeTensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [2, 1, 3],
        false
    );

    const inputB = makeTensor(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        [2, 1, 3],
        false
    );

    const [output1] = rnn.forward(inputA);
    rnn.forward(inputB);
    const [output2] = rnn.forward(inputA);

    assertArray(output1.shape, output2.shape);

    for (let i = 0; i < output1.numel; i++) {
        assert(
            Math.abs(output1.data[i] - output2.data[i]) < 1e-6,
            `Output differs at index ${i}`
        );
    }
});

test("RNN gradients flow through all timesteps", () => {
    const rnn = new RNN(2, 3);

    const input = makeTensor(
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        [3, 1, 2],
        true
    );

    const [output, hn] = rnn.forward(input);

    hn.sum().backward();

    // Every timestep's input should receive a gradient.
    for (let t = 0; t < 3; t++) {
        for (let f = 0; f < 2; f++) {
            const index = t * 2 + f;

            assert(
                Number.isFinite(input.grad.data[index]),
                `Invalid gradient at timestep ${t}, feature ${f}`
            );
        }
    }

    // Recurrent weights should receive gradients.
    assert(rnn.cell.h2h.w.grad !== null, "Missing recurrent weight gradient");
    assert(rnn.cell.x2h.w.grad !== null, "Missing input weight gradient");
});

test("RNN propagates hidden state with zero input", () => {
    const rnn = new RNN(2, 3);

    const input = makeTensor(
        new Float32Array(4),
        [2, 1, 2],
        false
    );

    const h0 = makeTensor(
        [0.5, -0.3, 0.8],
        [1, 3],
        false
    );

    const [output, hn] = rnn.forward(input, h0);

    assertArray(output.shape, [2, 1, 3]);
    assertArray(hn.shape, [1, 3]);

    assert(
        output.data.some(v => Math.abs(v) > 1e-6),
        "Hidden state should propagate through recurrent weights"
    );
});

test("RNN final hidden equals last output timestep", () => {
    const rnn = new RNN(2, 3);

    const input = makeTensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [3, 1, 2],
        false
    );

    const [output, hn] = rnn.forward(input);

    assertArray(output.shape, [3, 1, 3]);
    assertArray(hn.shape, [1, 3]);

    for (let h = 0; h < 3; h++) {
        assert(
            Math.abs(output.data[6 + h] - hn.data[h]) < 1e-6,
            `Mismatch in hidden feature ${h}`
        );
    }
});

test("RNN gradients from all output timesteps", () => {
    const rnn = new RNN(2, 3);

    const input = makeTensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [3, 1, 2],
        true
    );

    const [output] = rnn.forward(input);

    output.sum().backward();

    assert(
        input.grad !== null,
        "Input gradient should exist"
    );

    for (let i = 0; i < input.numel; i++) {
        assert(
            Number.isFinite(input.grad.data[i]),
            `Invalid input gradient at index ${i}`
        );
    }

    assert(
        rnn.cell.h2h.w.grad !== null,
        "Missing recurrent weight gradient"
    );

    assert(
        rnn.cell.x2h.w.grad !== null,
        "Missing input weight gradient"
    );
});

test("RNN input gradient numerical check", () => {
    const rnn = new RNN(2, 2);

    const values = [0.2, 0.4, 0.6, 0.8];
    const input = makeTensor(values, [2, 1, 2], true);

    const [output] = rnn.forward(input);
    output.sum().backward();

    const analytical = input.grad.data[0];

    const eps = 1e-3;

    function calculateLoss(firstValue) {
        const testValues = [...values];
        testValues[0] = firstValue;

        const testInput = makeTensor(testValues, [2, 1, 2], false);
        const [testOutput] = rnn.forward(testInput);

        return testOutput.data.reduce((sum, value) => sum + value, 0);
    }

    const numerical = (
        calculateLoss(values[0] + eps) -
        calculateLoss(values[0] - eps)
    ) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

test("RNN recurrent weight gradient numerical check", () => {
    const rnn = new RNN(2, 2);

    const values = [0.2, 0.4, 0.6, 0.8];
    const input = makeTensor(values, [2, 1, 2], false);

    const [output] = rnn.forward(input);
    output.sum().backward();

    const analytical = rnn.cell.h2h.w.grad.data[0];

    const eps = 1e-3;
    const weight = rnn.cell.h2h.w.data;
    const original = weight[0];

    function calculateLoss() {
        const [result] = rnn.forward(input);
        return result.data.reduce((sum, value) => sum + value, 0);
    }

    weight[0] = original + eps;
    const plus = calculateLoss();

    weight[0] = original - eps;
    const minus = calculateLoss();

    weight[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

test("RNN input weight gradient numerical check", () => {
    const rnn = new RNN(2, 2);

    const input = makeTensor(
        [0.2, 0.4, 0.6, 0.8],
        [2, 1, 2],
        false
    );

    const [output] = rnn.forward(input);
    output.sum().backward();

    const analytical = rnn.cell.x2h.w.grad.data[0];

    const eps = 1e-3;
    const weight = rnn.cell.x2h.w.data;
    const original = weight[0];

    function calculateLoss() {
        const [result] = rnn.forward(input);
        return result.data.reduce((sum, value) => sum + value, 0);
    }

    weight[0] = original + eps;
    const plus = calculateLoss();

    weight[0] = original - eps;
    const minus = calculateLoss();

    weight[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

test("RNN recurrent bias gradient numerical check", () => {
    const rnn = new RNN(2, 2);

    const input = makeTensor(
        [0.2, 0.4, 0.6, 0.8],
        [2, 1, 2],
        false
    );

    const [output] = rnn.forward(input);
    output.sum().backward();

    const analytical = rnn.cell.h2h.b.grad.data[0];

    const eps = 1e-3;
    const bias = rnn.cell.h2h.b.data;
    const original = bias[0];

    function calculateLoss() {
        const [result] = rnn.forward(input);
        return result.data.reduce((sum, value) => sum + value, 0);
    }

    bias[0] = original + eps;
    const plus = calculateLoss();

    bias[0] = original - eps;
    const minus = calculateLoss();

    bias[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

test("RNN input bias gradient numerical check", () => {
    const rnn = new RNN(2, 2);

    const input = makeTensor(
        [0.2, 0.4, 0.6, 0.8],
        [2, 1, 2],
        false
    );

    const [output] = rnn.forward(input);
    output.sum().backward();

    const analytical = rnn.cell.x2h.b.grad.data[0];

    const eps = 1e-3;
    const bias = rnn.cell.x2h.b.data;
    const original = bias[0];

    function calculateLoss() {
        const [result] = rnn.forward(input);
        return result.data.reduce((sum, value) => sum + value, 0);
    }

    bias[0] = original + eps;
    const plus = calculateLoss();

    bias[0] = original - eps;
    const minus = calculateLoss();

    bias[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

test("RNN initial hidden gradient numerical check", () => {
    const rnn = new RNN(2, 2);

    const input = makeTensor(
        [0.2, 0.4, 0.6, 0.8],
        [2, 1, 2],
        false
    );

    const h0 = makeTensor(
        [0.3, 0.5],
        [1, 2],
        true
    );

    const [output] = rnn.forward(input, h0);
    output.sum().backward();

    const analytical = h0.grad.data[0];

    const eps = 1e-3;
    const original = h0.data[0];

    function calculateLoss() {
        const [result] = rnn.forward(input, h0);
        return result.data.reduce((sum, value) => sum + value, 0);
    }

    h0.data[0] = original + eps;
    const plus = calculateLoss();

    h0.data[0] = original - eps;
    const minus = calculateLoss();

    h0.data[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

// 1. Forward output shape
test("GRUCell output shape", () => {
    const gru = new GRUCell(3, 4);

    const x = makeTensor(
        Array(6).fill(0.2), [2, 3], false
    );

    const h = makeTensor(
        Array(8).fill(0.3), [2, 4], false
    );

    const output = gru.forward(x, h);

    assert(
        JSON.stringify(output.shape) === JSON.stringify([2, 4]),
        `Unexpected shape: ${output.shape}`
    );
});

// 2. Forward output contains finite values
test("GRUCell output finite values", () => {
    const gru = new GRUCell(3, 4);

    const x = makeTensor(
        [0.1, 0.2, 0.3], [1, 3], false
    );

    const h = makeTensor(
        [0.2, 0.3, 0.4, 0.5], [1, 4], false
    );

    const output = gru.forward(x, h);

    for (const value of output.data) {
        assert(Number.isFinite(value), "Output contains non-finite value");
    }
});

// 3. Batch independence
test("GRUCell batch independence", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor(
        [0.1, 0.2, 0.3, 0.4], [2, 2], false
    );

    const h = makeTensor(
        [0.5, 0.6, 0.7, 0.8], [2, 2], false
    );

    const batched = gru.forward(x, h);

    const x0 = makeTensor([0.1, 0.2], [1, 2], false);
    const h0 = makeTensor([0.5, 0.6], [1, 2], false);

    const x1 = makeTensor([0.3, 0.4], [1, 2], false);
    const h1 = makeTensor([0.7, 0.8], [1, 2], false);

    const out0 = gru.forward(x0, h0);
    const out1 = gru.forward(x1, h1);

    for (let i = 0; i < 2; i++) {
        assert(Math.abs(batched.data[i] - out0.data[i]) < 1e-5);
        assert(Math.abs(batched.data[i + 2] - out1.data[i]) < 1e-5);
    }
});

// 4. Hidden state is not mutated
test("GRUCell does not mutate hidden state", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.1, 0.2], [1, 2], false);
    const h = makeTensor([0.3, 0.4], [1, 2], false);

    const original = Array.from(h.data);

    gru.forward(x, h);

    assert(
        h.data.every((v, i) => v === original[i]),
        "Hidden state was mutated"
    );
});

// 5. Gradients flow to input and hidden state
test("GRUCell backward through input and hidden state", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.1, 0.2], [1, 2], true);
    const h = makeTensor([0.3, 0.4], [1, 2], true);

    const output = gru.forward(x, h);
    output.sum().backward();

    for (const value of x.grad.data) {
        assert(Number.isFinite(value), "Invalid input gradient");
    }

    for (const value of h.grad.data) {
        assert(Number.isFinite(value), "Invalid hidden gradient");
    }
});

// 6. Gradients reach all six Linear layers
test("GRUCell gradients reach all Linear layers", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.1, 0.2], [1, 2], true);
    const h = makeTensor([0.3, 0.4], [1, 2], true);

    gru.forward(x, h).sum().backward();

    const layers = [
        gru.xz, gru.hz,
        gru.xr, gru.hr,
        gru.xn, gru.hn
    ];

    for (const layer of layers) {
        assert(layer.w.grad !== null, "Missing weight gradient");
        assert(layer.b.grad !== null, "Missing bias gradient");

        for (const value of layer.w.grad.data) {
            assert(Number.isFinite(value), "Invalid weight gradient");
        }

        for (const value of layer.b.grad.data) {
            assert(Number.isFinite(value), "Invalid bias gradient");
        }
    }
});

// 7. Gradients from candidate hidden state
test("GRUCell candidate path gradients", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.2, 0.4], [1, 2], true);
    const h = makeTensor([0.3, 0.5], [1, 2], true);

    gru.forward(x, h).sum().backward();

    assert(gru.xn.w.grad !== null);
    assert(gru.hn.w.grad !== null);
});

// 8. Gradients from update gate
test("GRUCell update gate gradients", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.2, 0.4], [1, 2], true);
    const h = makeTensor([0.3, 0.5], [1, 2], true);

    gru.forward(x, h).sum().backward();

    assert(gru.xz.w.grad !== null);
    assert(gru.hz.w.grad !== null);
});

// 9. Gradients from reset gate
test("GRUCell reset gate gradients", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.2, 0.4], [1, 2], true);
    const h = makeTensor([0.3, 0.5], [1, 2], true);

    gru.forward(x, h).sum().backward();

    assert(gru.xr.w.grad !== null);
    assert(gru.hr.w.grad !== null);
});

// 10. Numerical gradient check for input
test("GRUCell input gradient numerical check", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.2, 0.4], [1, 2], true);
    const h = makeTensor([0.3, 0.5], [1, 2], false);

    gru.forward(x, h).sum().backward();

    const analytical = x.grad.data[0];
    const original = x.data[0];
    const eps = 1e-3;

    function loss() {
        return gru.forward(x, h).data.reduce((a, b) => a + b, 0);
    }

    x.data[0] = original + eps;
    const plus = loss();

    x.data[0] = original - eps;
    const minus = loss();

    x.data[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

// 11. Numerical gradient check for hidden state
test("GRUCell hidden gradient numerical check", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.2, 0.4], [1, 2], false);
    const h = makeTensor([0.3, 0.5], [1, 2], true);

    gru.forward(x, h).sum().backward();

    const analytical = h.grad.data[0];
    const original = h.data[0];
    const eps = 1e-3;

    function loss() {
        return gru.forward(x, h).data.reduce((a, b) => a + b, 0);
    }

    h.data[0] = original + eps;
    const plus = loss();

    h.data[0] = original - eps;
    const minus = loss();

    h.data[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

// 12. Numerical gradient check for candidate recurrent weights
test("GRUCell candidate recurrent weight gradient numerical check", () => {
    const gru = new GRUCell(2, 2);

    const x = makeTensor([0.2, 0.4], [1, 2], false);
    const h = makeTensor([0.3, 0.5], [1, 2], false);

    gru.forward(x, h).sum().backward();

    const analytical = gru.hn.w.grad.data[0];
    const original = gru.hn.w.data[0];
    const eps = 1e-3;

    function loss() {
        return gru.forward(x, h).data.reduce((a, b) => a + b, 0);
    }

    gru.hn.w.data[0] = original + eps;
    const plus = loss();

    gru.hn.w.data[0] = original - eps;
    const minus = loss();

    gru.hn.w.data[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

// 13. Zero-sized batch
test("GRUCell zero-sized batch", () => {
    const gru = new GRUCell(3, 4);

    const x = makeTensor([], [0, 3], false);
    const h = makeTensor([], [0, 4], false);

    const output = gru.forward(x, h);

    assert(
        JSON.stringify(output.shape) === JSON.stringify([0, 4]),
        `Unexpected shape: ${output.shape}`
    );

    assert(output.numel === 0);
});

// 14. Bias disabled
test("GRUCell without bias", () => {
    const gru = new GRUCell(2, 2, false);

    const x = makeTensor([0.1, 0.2], [1, 2], true);
    const h = makeTensor([0.3, 0.4], [1, 2], true);

    const output = gru.forward(x, h);

    assert(JSON.stringify(output.shape) === JSON.stringify([1, 2]));

    output.sum().backward();

    const layers = [
        gru.xz, gru.hz,
        gru.xr, gru.hr,
        gru.xn, gru.hn
    ];

    for (const layer of layers) {
        assert(layer.b === null, "Bias should be disabled");
        assert(layer.w.grad !== null, "Weight gradient missing");
    }
});

// 1. Forward output shape
test("GRU output shape", () => {
    const gru = new GRU(4, 5);

    const input = makeTensor(
        Array(24).fill(0.2),
        [3, 2, 4],
        false
    );

    const [output, hn] = gru.forward(input);

    assert(
        JSON.stringify(output.shape) === JSON.stringify([3, 2, 5])
    );

    assert(
        JSON.stringify(hn.shape) === JSON.stringify([2, 5])
    );
});


// 2. Output contains finite values
test("GRU output finite values", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [3, 1, 2],
        false
    );

    const [output, hn] = gru.forward(input);

    for (const value of output.data) {
        assert(Number.isFinite(value));
    }

    for (const value of hn.data) {
        assert(Number.isFinite(value));
    }
});


// 3. Final hidden equals last output
test("GRU final hidden equals last output timestep", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        Array(12).fill(0.3),
        [3, 2, 2],
        false
    );

    const [output, hn] = gru.forward(input);

    const offset = (3 - 1) * 2 * 3;

    for (let i = 0; i < hn.numel; i++) {
        assert(
            Math.abs(hn.data[i] - output.data[offset + i]) < 1e-6,
            `Mismatch at ${i}: hn=${hn.data[i]}, output=${output.data[offset + i]}`
        );
    }
});


// 4. Explicit initial hidden state
test("GRU accepts explicit initial hidden state", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        Array(12).fill(0.2),
        [3, 2, 2],
        false
    );

    const h0 = makeTensor(
        Array(6).fill(0.4),
        [2, 3],
        false
    );

    const [output, hn] = gru.forward(input, h0);

    assert(JSON.stringify(output.shape) === JSON.stringify([3, 2, 3]));
    assert(JSON.stringify(hn.shape) === JSON.stringify([2, 3]));
});


// 5. Automatic hidden state initialization
test("GRU initializes zero hidden state", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        Array(12).fill(0.2),
        [3, 2, 2],
        false
    );

    const zeroH0 = makeTensor(
        Array(6).fill(0),
        [2, 3],
        false
    );

    const [automatic] = gru.forward(input);
    const [explicit] = gru.forward(input, zeroH0);

    for (let i = 0; i < automatic.numel; i++) {
        assert(
            Math.abs(automatic.data[i] - explicit.data[i]) < 1e-6
        );
    }
});


// 6. Batch independence
test("GRU batch independence", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        [
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2
        ],
        [3, 2, 2],
        false
    );

    const [batched] = gru.forward(input);

    for (let b = 0; b < 2; b++) {
        const single = input
            .slice(":", `${b}:${b + 1}`, ":");

        const [individual] = gru.forward(single);

        for (let t = 0; t < 3; t++) {
            for (let j = 0; j < 3; j++) {
                const batchedIndex = t * 6 + b * 3 + j;
                const individualIndex = t * 3 + j;

                assert(
                    Math.abs(
                        batched.data[batchedIndex] -
                        individual.data[individualIndex]
                    ) < 1e-5
                );
            }
        }
    }
});


// 7. Hidden state resets between forward calls
test("GRU resets hidden state between forward calls", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        Array(12).fill(0.2),
        [3, 2, 2],
        false
    );

    const [first] = gru.forward(input);
    const [second] = gru.forward(input);

    for (let i = 0; i < first.numel; i++) {
        assert(
            Math.abs(first.data[i] - second.data[i]) < 1e-6
        );
    }
});


// 8. Hidden state is not mutated
test("GRU does not mutate initial hidden state", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        Array(12).fill(0.2),
        [3, 2, 2],
        false
    );

    const h0 = makeTensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [2, 3],
        false
    );

    const original = Array.from(h0.data);

    gru.forward(input, h0);

    for (let i = 0; i < h0.numel; i++) {
        assert(h0.data[i] === original[i]);
    }
});


// 9. Non-contiguous input
test("GRU supports non-contiguous input", () => {
    const gru = new GRU(4, 5);

    const base = makeTensor(
        Array.from({ length: 24 }, (_, i) => i * 0.1),
        [2, 3, 4],
        false
    );

    const input = base.permute(1, 0, 2);

    assert(
        input.isContiguous() === false,
        "Expected non-contiguous input"
    );

    const values = [];

    for (let t = 0; t < 3; t++) {
        for (let b = 0; b < 2; b++) {
            for (let f = 0; f < 4; f++) {
                const index = (b * 3 + t) * 4 + f;
                values.push(base.data[index]);
            }
        }
    }

    const reference = makeTensor(values, [3, 2, 4], false);

    assert(reference.isContiguous(), "Reference should be contiguous");

    const [output1, hn1] = gru.forward(input);
    const [output2, hn2] = gru.forward(reference);

    for (let i = 0; i < output1.numel; i++) {
        assert(
            Math.abs(output1.data[i] - output2.data[i]) < 1e-5,
            `Output mismatch at ${i}`
        );
    }

    for (let i = 0; i < hn1.numel; i++) {
        assert(
            Math.abs(hn1.data[i] - hn2.data[i]) < 1e-5,
            `Hidden mismatch at ${i}`
        );
    }
});


// 10. Backward through all timesteps
test("GRU backward through all timesteps", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        Array(12).fill(0.2),
        [3, 2, 2],
        true
    );

    const [output] = gru.forward(input);

    output.sum().backward();

    for (const value of input.grad.data) {
        assert(Number.isFinite(value));
    }

    const layers = [
        gru.cell.xz,
        gru.cell.hz,
        gru.cell.xr,
        gru.cell.hr,
        gru.cell.xn,
        gru.cell.hn
    ];

    for (const layer of layers) {
        assert(layer.w.grad !== null);
        assert(layer.b.grad !== null);
    }
});


// 11. Gradient flows into initial hidden state
test("GRU initial hidden state gradient", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        Array(12).fill(0.2),
        [3, 2, 2],
        false
    );

    const h0 = makeTensor(
        Array(6).fill(0.3),
        [2, 3],
        true
    );

    const [output] = gru.forward(input, h0);

    output.sum().backward();

    for (const value of h0.grad.data) {
        assert(Number.isFinite(value));
    }
});


// 12. Final hidden state backward
test("GRU final hidden backward", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        Array(12).fill(0.2),
        [3, 2, 2],
        true
    );

    const h0 = makeTensor(
        Array(6).fill(0.3),
        [2, 3],
        true
    );

    const [, hn] = gru.forward(input, h0);

    hn.sum().backward();

    for (const value of input.grad.data) {
        assert(Number.isFinite(value));
    }

    for (const value of h0.grad.data) {
        assert(Number.isFinite(value));
    }
});


// 13. Empty sequence
test("GRU empty sequence", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor([], [0, 2, 2], false);

    const [output, hn] = gru.forward(input);

    assert(JSON.stringify(output.shape) === JSON.stringify([0, 2, 3]));
    assert(JSON.stringify(hn.shape) === JSON.stringify([2, 3]));

    assert(output.numel === 0);

    for (const value of hn.data) {
        assert(value === 0);
    }
});


// 14. Empty sequence with explicit hidden state
test("GRU empty sequence preserves initial hidden state", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor([], [0, 2, 2], false);

    const h0 = makeTensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [2, 3],
        false
    );

    const [output, hn] = gru.forward(input, h0);

    assert(output.numel === 0);

    for (let i = 0; i < h0.numel; i++) {
        assert(hn.data[i] === h0.data[i]);
    }
});


// 15. Zero-sized batch
test("GRU zero-sized batch", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor([], [3, 0, 2], false);

    const [output, hn] = gru.forward(input);

    assert(JSON.stringify(output.shape) === JSON.stringify([3, 0, 3]));
    assert(JSON.stringify(hn.shape) === JSON.stringify([0, 3]));

    assert(output.numel === 0);
    assert(hn.numel === 0);
});


// 16. Sequence length one
test("GRU sequence length one", () => {
    const gru = new GRU(2, 3);

    const input = makeTensor(
        [0.1, 0.2, 0.3, 0.4],
        [1, 2, 2],
        false
    );

    const h0 = makeTensor(
        Array(6).fill(0.2),
        [2, 3],
        false
    );

    const [output, hn] = gru.forward(input, h0);

    const x = input.squeeze(0);
    const expected = gru.cell.forward(x, h0);

    assert(JSON.stringify(output.shape) === JSON.stringify([1, 2, 3]));

    for (let i = 0; i < hn.numel; i++) {
        assert(Math.abs(hn.data[i] - expected.data[i]) < 1e-6);
    }
});


// 17. Numerical gradient check for input
test("GRU input gradient numerical check", () => {
    const gru = new GRU(2, 2);

    const input = makeTensor(
        [0.2, 0.4, 0.6, 0.8],
        [2, 1, 2],
        true
    );

    const [output] = gru.forward(input);
    output.sum().backward();

    const analytical = input.grad.data[0];
    const original = input.data[0];
    const eps = 1e-3;

    function loss() {
        return gru.forward(input)[0].data.reduce((a, b) => a + b, 0);
    }

    input.data[0] = original + eps;
    const plus = loss();

    input.data[0] = original - eps;
    const minus = loss();

    input.data[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});


// 18. Numerical gradient check for initial hidden state
test("GRU initial hidden gradient numerical check", () => {
    const gru = new GRU(2, 2);

    const input = makeTensor(
        [0.2, 0.4, 0.6, 0.8],
        [2, 1, 2],
        false
    );

    const h0 = makeTensor(
        [0.3, 0.5],
        [1, 2],
        true
    );

    const [output] = gru.forward(input, h0);
    output.sum().backward();

    const analytical = h0.grad.data[0];
    const original = h0.data[0];
    const eps = 1e-3;

    function loss() {
        return gru.forward(input, h0)[0].data.reduce((a, b) => a + b, 0);
    }

    h0.data[0] = original + eps;
    const plus = loss();

    h0.data[0] = original - eps;
    const minus = loss();

    h0.data[0] = original;

    const numerical = (plus - minus) / (2 * eps);

    assert(
        Math.abs(analytical - numerical) < 1e-2,
        `Analytical: ${analytical}, Numerical: ${numerical}`
    );
});

// --------------------------------------------------
// 1. Forward output shapes
// --------------------------------------------------

test("LSTMCell forward output shapes", () => {
    const cell = new LSTMCell(4, 5);

    const x = makeTensor(Array(12).fill(0.2), [3, 4]);
    const h = makeTensor(Array(15).fill(0.1), [3, 5]);
    const c = makeTensor(Array(15).fill(0.3), [3, 5]);

    const [hn, cn] = cell.forward(x, h, c);

    assert(
        hn.shape.join(",") === "3,5",
        "Incorrect hidden shape"
    );

    assert(
        cn.shape.join(",") === "3,5",
        "Incorrect cell shape"
    );
});


// --------------------------------------------------
// 2. Output values must be finite
// --------------------------------------------------

test("LSTMCell outputs finite values", () => {
    const cell = new LSTMCell(3, 4);

    const x = makeTensor(Array(6).fill(0.5), [2, 3]);
    const h = makeTensor(Array(8).fill(0.2), [2, 4]);
    const c = makeTensor(Array(8).fill(0.4), [2, 4]);

    const [hn, cn] = cell.forward(x, h, c);

    for (const value of hn.data) {
        assert(Number.isFinite(value), "Invalid hidden value");
    }

    for (const value of cn.data) {
        assert(Number.isFinite(value), "Invalid cell value");
    }
});


// --------------------------------------------------
// 3. Hidden state and cell state must not mutate
// --------------------------------------------------

test("LSTMCell does not mutate input states", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([0.1, 0.2], [1, 2]);
    const h = makeTensor([0.3, 0.4, 0.5], [1, 3]);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3]);

    const hBefore = Array.from(h.data);
    const cBefore = Array.from(c.data);

    cell.forward(x, h, c);

    assert(
        h.data.every((v, i) => v === hBefore[i]),
        "Hidden state was mutated"
    );

    assert(
        c.data.every((v, i) => v === cBefore[i]),
        "Cell state was mutated"
    );
});


// --------------------------------------------------
// 4. Batch independence
// --------------------------------------------------

test("LSTMCell batch independence", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([
        0.1, 0.2,
        0.3, 0.4
    ], [2, 2]);

    const h = makeTensor([
        0.2, 0.3, 0.4,
        0.5, 0.6, 0.7
    ], [2, 3]);

    const c = makeTensor([
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6
    ], [2, 3]);

    const [hn, cn] = cell.forward(x, h, c);

    for (let b = 0; b < 2; b++) {
        const xSingle = makeTensor(
            x.data.slice(b * 2, (b + 1) * 2),
            [1, 2]
        );

        const hSingle = makeTensor(
            h.data.slice(b * 3, (b + 1) * 3),
            [1, 3]
        );

        const cSingle = makeTensor(
            c.data.slice(b * 3, (b + 1) * 3),
            [1, 3]
        );

        const [hs, cs] = cell.forward(
            xSingle,
            hSingle,
            cSingle
        );

        for (let j = 0; j < 3; j++) {
            assert(
                Math.abs(hn.data[b * 3 + j] - hs.data[j]) < 1e-6,
                "Hidden batch independence failed"
            );

            assert(
                Math.abs(cn.data[b * 3 + j] - cs.data[j]) < 1e-6,
                "Cell batch independence failed"
            );
        }
    }
});


// --------------------------------------------------
// 5. Backward through input, hidden and cell states
// --------------------------------------------------

test("LSTMCell backward through all inputs", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([0.1, 0.2], [1, 2], true);
    const h = makeTensor([0.3, 0.4, 0.5], [1, 3], true);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3], true);

    const [hn, cn] = cell.forward(x, h, c);

    const loss = hn.sum().add(cn.sum());

    loss.backward();

    assert(x.grad !== null, "Input gradient missing");
    assert(h.grad !== null, "Hidden gradient missing");
    assert(c.grad !== null, "Cell gradient missing");
});


// --------------------------------------------------
// 6. Gradients reach all eight Linear layers
// --------------------------------------------------

test("LSTMCell gradients reach all Linear layers", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([0.1, 0.2], [1, 2], true);
    const h = makeTensor([0.3, 0.4, 0.5], [1, 3], true);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3], true);

    const [hn, cn] = cell.forward(x, h, c);

    hn.sum().add(cn.sum()).backward();

    const layers = [
        cell.xf, cell.hf,
        cell.xi, cell.hi,
        cell.xc, cell.hc,
        cell.xo, cell.ho
    ];

    for (let i = 0; i < layers.length; i++) {
        const params = layers[i].parameters();

        assert(params.length > 0, `Layer ${i} has no parameters`);

        for (const param of params) {
            assert(param.grad !== null, `Layer ${i} gradient missing`);
        }
    }
});


// --------------------------------------------------
// 7. Forget gate gradient path
// --------------------------------------------------

test("LSTMCell forget gate receives gradients", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([0.2, 0.4], [1, 2], true);
    const h = makeTensor([0.1, 0.3, 0.5], [1, 3], true);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3], true);

    const [, cn] = cell.forward(x, h, c);

    cn.sum().backward();

    for (const layer of [cell.xf, cell.hf]) {
        for (const p of layer.parameters()) {
            assert(p.grad !== null, "Forget gate gradient missing");
        }
    }
});


// --------------------------------------------------
// 8. Input gate gradient path
// --------------------------------------------------

test("LSTMCell input gate receives gradients", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([0.2, 0.4], [1, 2], true);
    const h = makeTensor([0.1, 0.3, 0.5], [1, 3], true);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3], true);

    const [, cn] = cell.forward(x, h, c);

    cn.sum().backward();

    for (const layer of [cell.xi, cell.hi]) {
        for (const p of layer.parameters()) {
            assert(p.grad !== null, "Input gate gradient missing");
        }
    }
});


// --------------------------------------------------
// 9. Candidate cell gradient path
// --------------------------------------------------

test("LSTMCell candidate receives gradients", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([0.2, 0.4], [1, 2], true);
    const h = makeTensor([0.1, 0.3, 0.5], [1, 3], true);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3], true);

    const [, cn] = cell.forward(x, h, c);

    cn.sum().backward();

    for (const layer of [cell.xc, cell.hc]) {
        for (const p of layer.parameters()) {
            assert(p.grad !== null, "Candidate gradient missing");
        }
    }
});


// --------------------------------------------------
// 10. Output gate gradient path
// --------------------------------------------------

test("LSTMCell output gate receives gradients", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([0.2, 0.4], [1, 2], true);
    const h = makeTensor([0.1, 0.3, 0.5], [1, 3], true);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3], true);

    const [hn] = cell.forward(x, h, c);

    hn.sum().backward();

    for (const layer of [cell.xo, cell.ho]) {
        for (const p of layer.parameters()) {
            assert(p.grad !== null, "Output gate gradient missing");
        }
    }
});


// --------------------------------------------------
// 11. Numerical gradient: input
// --------------------------------------------------

test("LSTMCell numerical gradient input", () => {
    const cell = new LSTMCell(2, 2);

    const x = makeTensor([0.2, -0.3], [1, 2], true);
    const h = makeTensor([0.1, 0.4], [1, 2], true);
    const c = makeTensor([0.5, -0.2], [1, 2], true);

    const [hn, cn] = cell.forward(x, h, c);

    hn.sum().add(cn.sum()).backward();

    const eps = 1e-3;

    for (let i = 0; i < x.numel; i++) {
        const original = x.data[i];

        x.data[i] = original + eps;
        const [hp, cp] = cell.forward(x, h, c);
        const plus = hp.data.reduce((a, b) => a + b, 0)
            + cp.data.reduce((a, b) => a + b, 0);

        x.data[i] = original - eps;
        const [hm, cm] = cell.forward(x, h, c);
        const minus = hm.data.reduce((a, b) => a + b, 0)
            + cm.data.reduce((a, b) => a + b, 0);

        x.data[i] = original;

        const numerical = (plus - minus) / (2 * eps);

        assert(
            Math.abs(x.grad.data[i] - numerical) < 2e-2,
            `Input gradient mismatch at ${i}: ${x.grad.data[i]} vs ${numerical}`
        );
    }
});


// --------------------------------------------------
// 12. Numerical gradient: hidden state
// --------------------------------------------------

test("LSTMCell numerical gradient hidden state", () => {
    const cell = new LSTMCell(2, 2);

    const x = makeTensor([0.2, -0.3], [1, 2], true);
    const h = makeTensor([0.1, 0.4], [1, 2], true);
    const c = makeTensor([0.5, -0.2], [1, 2], true);

    const [hn, cn] = cell.forward(x, h, c);

    hn.sum().add(cn.sum()).backward();

    const eps = 1e-3;

    for (let i = 0; i < h.numel; i++) {
        const original = h.data[i];

        h.data[i] = original + eps;
        const [hp, cp] = cell.forward(x, h, c);
        const plus = hp.data.reduce((a, b) => a + b, 0)
            + cp.data.reduce((a, b) => a + b, 0);

        h.data[i] = original - eps;
        const [hm, cm] = cell.forward(x, h, c);
        const minus = hm.data.reduce((a, b) => a + b, 0)
            + cm.data.reduce((a, b) => a + b, 0);

        h.data[i] = original;

        const numerical = (plus - minus) / (2 * eps);

        assert(
            Math.abs(h.grad.data[i] - numerical) < 2e-2,
            `Hidden gradient mismatch at ${i}: ${h.grad.data[i]} vs ${numerical}`
        );
    }
});


// --------------------------------------------------
// 13. Numerical gradient: cell state
// --------------------------------------------------

test("LSTMCell numerical gradient cell state", () => {
    const cell = new LSTMCell(2, 2);

    const x = makeTensor([0.2, -0.3], [1, 2], true);
    const h = makeTensor([0.1, 0.4], [1, 2], true);
    const c = makeTensor([0.5, -0.2], [1, 2], true);

    const [hn, cn] = cell.forward(x, h, c);

    hn.sum().add(cn.sum()).backward();

    const eps = 1e-3;

    for (let i = 0; i < c.numel; i++) {
        const original = c.data[i];

        c.data[i] = original + eps;
        const [hp, cp] = cell.forward(x, h, c);
        const plus = hp.data.reduce((a, b) => a + b, 0)
            + cp.data.reduce((a, b) => a + b, 0);

        c.data[i] = original - eps;
        const [hm, cm] = cell.forward(x, h, c);
        const minus = hm.data.reduce((a, b) => a + b, 0)
            + cm.data.reduce((a, b) => a + b, 0);

        c.data[i] = original;

        const numerical = (plus - minus) / (2 * eps);

        assert(
            Math.abs(c.grad.data[i] - numerical) < 2e-2,
            `Cell gradient mismatch at ${i}: ${c.grad.data[i]} vs ${numerical}`
        );
    }
});


// --------------------------------------------------
// 14. Sequence-like repeated calls
// --------------------------------------------------

test("LSTMCell supports repeated timestep calls", () => {
    const cell = new LSTMCell(2, 3);

    let h = makeTensor([0, 0, 0], [1, 3]);
    let c = makeTensor([0, 0, 0], [1, 3]);

    for (let t = 0; t < 5; t++) {
        const x = makeTensor(
            [0.1 * (t + 1), 0.2 * (t + 1)],
            [1, 2]
        );

        [h, c] = cell.forward(x, h, c);
    }

    assert(h.shape.join(",") === "1,3", "Invalid hidden shape");
    assert(c.shape.join(",") === "1,3", "Invalid cell shape");

    for (const v of h.data) {
        assert(Number.isFinite(v), "Invalid final hidden value");
    }

    for (const v of c.data) {
        assert(Number.isFinite(v), "Invalid final cell value");
    }
});


// --------------------------------------------------
// 15. Zero-sized batch
// --------------------------------------------------

test("LSTMCell supports zero-sized batch", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([], [0, 2]);
    const h = makeTensor([], [0, 3]);
    const c = makeTensor([], [0, 3]);

    const [hn, cn] = cell.forward(x, h, c);

    assert(hn.shape.join(",") === "0,3", "Invalid hidden shape");
    assert(cn.shape.join(",") === "0,3", "Invalid cell shape");
});


// --------------------------------------------------
// 16. Bias disabled
// --------------------------------------------------

test("LSTMCell supports bias disabled", () => {
    const cell = new LSTMCell(2, 3, false);

    const x = makeTensor([0.1, 0.2], [1, 2]);
    const h = makeTensor([0.3, 0.4, 0.5], [1, 3]);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3]);

    const [hn, cn] = cell.forward(x, h, c);

    assert(hn.shape.join(",") === "1,3", "Invalid hidden shape");
    assert(cn.shape.join(",") === "1,3", "Invalid cell shape");

    for (const layer of [
        cell.xf, cell.hf,
        cell.xi, cell.hi,
        cell.xc, cell.hc,
        cell.xo, cell.ho
    ]) {
        assert(
            layer.bias === null || layer.bias === undefined,
            "Bias should be disabled"
        );
    }
});


// --------------------------------------------------
// 17. Different hidden and input sizes
// --------------------------------------------------

test("LSTMCell supports different input and hidden sizes", () => {
    const cell = new LSTMCell(4, 2);

    const x = makeTensor([0.1, 0.2, 0.3, 0.4], [1, 4]);
    const h = makeTensor([0.5, 0.6], [1, 2]);
    const c = makeTensor([0.7, 0.8], [1, 2]);

    const [hn, cn] = cell.forward(x, h, c);

    assert(hn.shape.join(",") === "1,2", "Invalid hidden shape");
    assert(cn.shape.join(",") === "1,2", "Invalid cell shape");
});


// --------------------------------------------------
// 18. Hidden and cell outputs are distinct
// --------------------------------------------------

test("LSTMCell returns separate hidden and cell states", () => {
    const cell = new LSTMCell(2, 3);

    const x = makeTensor([0.2, 0.4], [1, 2]);
    const h = makeTensor([0.1, 0.3, 0.5], [1, 3]);
    const c = makeTensor([0.6, 0.7, 0.8], [1, 3]);

    const result = cell.forward(x, h, c);

    assert(Array.isArray(result), "Expected array result");
    assert(result.length === 2, "Expected two outputs");

    const [hn, cn] = result;

    assert(hn !== cn, "Hidden and cell states must be separate tensors");
    assert(hn.shape.join(",") === cn.shape.join(","), "State shapes differ");
});

function allFinite(tensor) {
    return tensor.data.every(Number.isFinite);
}

function close(a, b, tolerance = 1e-5) {
    return Math.abs(a - b) < tolerance;
}

// 1. Output shapes
test("LSTM output shapes", () => {
    const lstm = new LSTM(3, 4);

    const input = makeTensor(
        Array(5 * 2 * 3).fill(0.5),
        [5, 2, 3]
    );

    const [output, hn, cn] = lstm.forward(input);

    assert(
        JSON.stringify(output.shape) === JSON.stringify([5, 2, 4]),
        "Incorrect output shape"
    );

    assert(
        JSON.stringify(hn.shape) === JSON.stringify([2, 4]),
        "Incorrect final hidden shape"
    );

    assert(
        JSON.stringify(cn.shape) === JSON.stringify([2, 4]),
        "Incorrect final cell shape"
    );
});

// 2. Finite outputs
test("LSTM outputs are finite", () => {
    const lstm = new LSTM(3, 4);

    const input = makeTensor(
        Array(5 * 2 * 3).fill(0.2),
        [5, 2, 3]
    );

    const [output, hn, cn] = lstm.forward(input);

    assert(allFinite(output), "Output contains non-finite values");
    assert(allFinite(hn), "Hidden state contains non-finite values");
    assert(allFinite(cn), "Cell state contains non-finite values");
});

// 3. Zero initial states
test("LSTM works with omitted initial states", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(4 * 2 * 2).fill(0.1),
        [4, 2, 2]
    );

    const [output, hn, cn] = lstm.forward(input);

    assert(output.shape[0] === 4, "Incorrect sequence length");
    assert(hn.shape[0] === 2, "Incorrect batch size");
    assert(cn.shape[0] === 2, "Incorrect batch size");
});

// 4. Explicit initial states
test("LSTM accepts explicit initial states", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(4 * 2 * 2).fill(0.1),
        [4, 2, 2]
    );

    const h0 = makeTensor(Array(6).fill(0.3), [2, 3]);
    const c0 = makeTensor(Array(6).fill(0.4), [2, 3]);

    const [output, hn, cn] = lstm.forward(input, h0, c0);

    assert(allFinite(output), "Invalid output");
    assert(allFinite(hn), "Invalid final hidden state");
    assert(allFinite(cn), "Invalid final cell state");
});

// 5. Final states equal last output timestep
test("Final hidden state equals last output timestep", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(4 * 2 * 2).fill(0.2),
        [4, 2, 2]
    );

    const [output, hn] = lstm.forward(input);

    const lastOffset = 3 * 2 * 3;

    for (let i = 0; i < hn.data.length; i++) {
        assert(
            close(output.data[lastOffset + i], hn.data[i]),
            `Mismatch at ${i}`
        );
    }
});

// 6. Inputs are not mutated
test("LSTM does not mutate input or initial states", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.2),
        [3, 2, 2]
    );

    const h0 = makeTensor(Array(6).fill(0.1), [2, 3]);
    const c0 = makeTensor(Array(6).fill(0.2), [2, 3]);

    const inputBefore = values(input);
    const hBefore = values(h0);
    const cBefore = values(c0);

    lstm.forward(input, h0, c0);

    assert(
        values(input).every((v, i) => v === inputBefore[i]),
        "Input was mutated"
    );

    assert(
        values(h0).every((v, i) => v === hBefore[i]),
        "Initial hidden state was mutated"
    );

    assert(
        values(c0).every((v, i) => v === cBefore[i]),
        "Initial cell state was mutated"
    );
});

// 7. Backward through input
test("LSTM propagates gradients to input", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.2),
        [3, 2, 2],
        true
    );

    const [output] = lstm.forward(input);

    output.sum().backward();

    assert(input.grad != null, "Input gradient missing");
    assert(
        input.grad.data.some(v => Math.abs(v) > 1e-8),
        "Input gradient is zero"
    );
});

// 8. Backward through initial hidden state
test("LSTM propagates gradients to h0", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.2),
        [3, 2, 2]
    );

    const h0 = makeTensor(Array(6).fill(0.1), [2, 3], true);
    const c0 = makeTensor(Array(6).fill(0.2), [2, 3]);

    const [output] = lstm.forward(input, h0, c0);

    output.sum().backward();

    assert(h0.grad != null, "h0 gradient missing");
    assert(
        h0.grad.data.some(v => Math.abs(v) > 1e-8),
        "h0 gradient is zero"
    );
});

// 9. Backward through initial cell state
test("LSTM propagates gradients to c0", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.2),
        [3, 2, 2]
    );

    const h0 = makeTensor(Array(6).fill(0.1), [2, 3]);
    const c0 = makeTensor(Array(6).fill(0.2), [2, 3], true);

    const [output] = lstm.forward(input, h0, c0);

    output.sum().backward();

    assert(c0.grad != null, "c0 gradient missing");
    assert(
        c0.grad.data.some(v => Math.abs(v) > 1e-8),
        "c0 gradient is zero"
    );
});

// 10. Gradients reach all parameters
test("LSTM propagates gradients to all parameters", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(4 * 2 * 2).fill(0.2),
        [4, 2, 2],
        true
    );

    const [output] = lstm.forward(input);

    output.sum().backward();

    const parameters = lstm.parameters();

    assert(parameters.length > 0, "No parameters found");

    for (const parameter of parameters) {
        assert(parameter.grad != null, "Parameter gradient missing");
    }
});

// 11. Repeated calls
test("LSTM can process multiple sequences", () => {
    const lstm = new LSTM(2, 3);

    const input1 = makeTensor(
        Array(3 * 2 * 2).fill(0.1),
        [3, 2, 2]
    );

    const input2 = makeTensor(
        Array(3 * 2 * 2).fill(0.3),
        [3, 2, 2]
    );

    const [out1] = lstm.forward(input1);
    const [out2] = lstm.forward(input2);

    assert(allFinite(out1), "First output invalid");
    assert(allFinite(out2), "Second output invalid");
});

// 12. Empty sequence
test("LSTM handles empty sequence", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor([], [0, 2, 2]);

    const [output, hn, cn] = lstm.forward(input);

    assert(
        JSON.stringify(output.shape) === JSON.stringify([0, 2, 3]),
        "Incorrect empty output shape"
    );

    assert(hn.shape[0] === 2, "Incorrect hidden state shape");
    assert(cn.shape[0] === 2, "Incorrect cell state shape");
});

// 13. Bias disabled
test("LSTM works without bias", () => {
    const lstm = new LSTM(2, 3, false);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.2),
        [3, 2, 2]
    );

    const [output, hn, cn] = lstm.forward(input);

    assert(allFinite(output), "Invalid output");
    assert(allFinite(hn), "Invalid hidden state");
    assert(allFinite(cn), "Invalid cell state");
});

// 14. Different input and hidden sizes
test("LSTM supports different input and hidden sizes", () => {
    const lstm = new LSTM(5, 2);

    const input = makeTensor(
        Array(4 * 3 * 5).fill(0.2),
        [4, 3, 5]
    );

    const [output, hn, cn] = lstm.forward(input);

    assert(
        JSON.stringify(output.shape) === JSON.stringify([4, 3, 2]),
        "Incorrect output shape"
    );

    assert(
        JSON.stringify(hn.shape) === JSON.stringify([3, 2]),
        "Incorrect hidden shape"
    );

    assert(
        JSON.stringify(cn.shape) === JSON.stringify([3, 2]),
        "Incorrect cell shape"
    );
});

function assertShape(tensor, expected, message) {
    assert(
        JSON.stringify(tensor.shape) === JSON.stringify(expected),
        message
    );
}

function objective(lstm, input, h0, c0) {
    const [output, hn, cn] = lstm.forward(input, h0, c0);

    return (
        output.sum()
            .add(hn.sum())
            .add(cn.sum())
            .data[0]
    );
}

function numericalGradient(data, index, evaluate, epsilon = 1e-3) {
    const original = data[index];

    data[index] = original + epsilon;
    const positive = evaluate();

    data[index] = original - epsilon;
    const negative = evaluate();

    data[index] = original;

    return (positive - negative) / (2 * epsilon);
}

function checkNumericalGradient(
    tensor,
    index,
    analytical,
    evaluate,
    tolerance = 2e-2
) {
    const numerical = numericalGradient(
        tensor.data,
        index,
        evaluate
    );

    assert(
        Math.abs(analytical - numerical) < tolerance,
        `Gradient mismatch at ${index}: ${analytical} vs ${numerical}`
    );
}

// --------------------------------------------------
// 1. Forward tests
// --------------------------------------------------

test("LSTM output shapes", () => {
    const lstm = new LSTM(3, 4);

    const input = makeTensor(
        Array(5 * 2 * 3).fill(0.2),
        [5, 2, 3]
    );

    const [output, hn, cn] = lstm.forward(input);

    assertShape(output, [5, 2, 4], "Wrong output shape");
    assertShape(hn, [2, 4], "Wrong hidden shape");
    assertShape(cn, [2, 4], "Wrong cell shape");
});

test("LSTM outputs finite values", () => {
    const lstm = new LSTM(3, 4);

    const input = makeTensor(
        Array(4 * 2 * 3).fill(0.3),
        [4, 2, 3]
    );

    const [output, hn, cn] = lstm.forward(input);

    assert(allFinite(output), "Output contains non-finite values");
    assert(allFinite(hn), "Hidden state contains non-finite values");
    assert(allFinite(cn), "Cell state contains non-finite values");
});

test("LSTM sequence length one", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        [0.1, 0.2, 0.3, 0.4],
        [1, 2, 2]
    );

    const [output, hn] = lstm.forward(input);

    assertShape(output, [1, 2, 3], "Wrong output shape");
    
    for (let i = 0; i < hn.data.length; i++) {
        assert(
            close(output.data[i], hn.data[i]),
            `Final hidden mismatch at ${i}`
        );
    }
});

test("LSTM final hidden equals last output timestep", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(4 * 2 * 2).fill(0.2),
        [4, 2, 2]
    );

    const [output, hn] = lstm.forward(input);

    const lastOffset = 3 * 2 * 3;

    for (let i = 0; i < hn.data.length; i++) {
        assert(
            close(output.data[lastOffset + i], hn.data[i]),
            `Final hidden mismatch at ${i}`
        );
    }
});

// --------------------------------------------------
// 2. Initial state tests
// --------------------------------------------------

test("LSTM initializes zero states automatically", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.1),
        [3, 2, 2]
    );

    const [output, hn, cn] = lstm.forward(input);

    assert(allFinite(output), "Invalid output");
    assert(allFinite(hn), "Invalid hidden state");
    assert(allFinite(cn), "Invalid cell state");
});

test("LSTM accepts explicit initial states", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.2),
        [3, 2, 2]
    );

    const h0 = makeTensor(Array(6).fill(0.3), [2, 3]);
    const c0 = makeTensor(Array(6).fill(0.4), [2, 3]);

    const [output, hn, cn] = lstm.forward(input, h0, c0);

    assert(allFinite(output), "Invalid output");
    assert(allFinite(hn), "Invalid hidden state");
    assert(allFinite(cn), "Invalid cell state");
});

test("LSTM preserves initial states when sequence is empty", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor([], [0, 2, 2]);

    const h0 = makeTensor(Array(6).fill(0.3), [2, 3]);
    const c0 = makeTensor(Array(6).fill(0.4), [2, 3]);

    const [output, hn, cn] = lstm.forward(input, h0, c0);

    assertShape(output, [0, 2, 3], "Wrong empty output shape");

    for (let i = 0; i < h0.data.length; i++) {
        assert(close(hn.data[i], h0.data[i]), "h0 was not preserved");
        assert(close(cn.data[i], c0.data[i]), "c0 was not preserved");
    }
});

// --------------------------------------------------
// 3. Input and state mutation tests
// --------------------------------------------------

test("LSTM does not mutate input", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.2),
        [3, 2, 2]
    );

    const before = Array.from(input.data);

    lstm.forward(input);

    assert(
        input.data.every((v, i) => v === before[i]),
        "Input was mutated"
    );
});

test("LSTM does not mutate initial states", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 2 * 2).fill(0.2),
        [3, 2, 2]
    );

    const h0 = makeTensor(Array(6).fill(0.3), [2, 3]);
    const c0 = makeTensor(Array(6).fill(0.4), [2, 3]);

    const hBefore = Array.from(h0.data);
    const cBefore = Array.from(c0.data);

    lstm.forward(input, h0, c0);

    assert(
        h0.data.every((v, i) => v === hBefore[i]),
        "h0 was mutated"
    );

    assert(
        c0.data.every((v, i) => v === cBefore[i]),
        "c0 was mutated"
    );
});

// --------------------------------------------------
// 4. Batch and sequence behavior
// --------------------------------------------------

test("LSTM supports different input and hidden sizes", () => {
    const lstm = new LSTM(5, 2);

    const input = makeTensor(
        Array(4 * 3 * 5).fill(0.2),
        [4, 3, 5]
    );

    const [output, hn, cn] = lstm.forward(input);

    assertShape(output, [4, 3, 2], "Wrong output shape");
    assertShape(hn, [3, 2], "Wrong hidden shape");
    assertShape(cn, [3, 2], "Wrong cell shape");
});

test("LSTM supports zero-sized batch", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor([], [4, 0, 2]);

    const [output, hn, cn] = lstm.forward(input);

    assertShape(output, [4, 0, 3], "Wrong output shape");
    assertShape(hn, [0, 3], "Wrong hidden shape");
    assertShape(cn, [0, 3], "Wrong cell shape");
});

test("LSTM supports non-contiguous input", () => {
    const lstm = new LSTM(2, 3);

    const base = makeTensor(
        Array(3 * 2 * 4).fill(0.2),
        [3, 2, 4]
    );

    const input = base.permute(0, 2, 1);

    assert(!input.isContiguous(), "Input should be non-contiguous");

    const [output, hn, cn] = lstm.forward(input);

    assertShape(output, [3, 4, 3], "Wrong output shape");
    assert(allFinite(output), "Invalid output");
    assert(allFinite(hn), "Invalid hidden state");
    assert(allFinite(cn), "Invalid cell state");
});

test("LSTM resets states between forward calls", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 1 * 2).fill(0.2),
        [3, 1, 2]
    );

    const [output1] = lstm.forward(input);
    const [output2] = lstm.forward(input);

    for (let i = 0; i < output1.data.length; i++) {
        assert(
            close(output1.data[i], output2.data[i]),
            `Output differs at ${i}`
        );
    }
});

test("LSTM can process different sequences repeatedly", () => {
    const lstm = new LSTM(2, 3);

    const input1 = makeTensor(
        Array(3 * 1 * 2).fill(0.1),
        [3, 1, 2]
    );

    const input2 = makeTensor(
        Array(3 * 1 * 2).fill(0.5),
        [3, 1, 2]
    );

    const [out1] = lstm.forward(input1);
    const [out2] = lstm.forward(input2);

    assert(allFinite(out1), "First output invalid");
    assert(allFinite(out2), "Second output invalid");
});

// --------------------------------------------------
// 5. Backward / BPTT tests
// --------------------------------------------------

test("LSTM propagates gradients to input", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 1 * 2).fill(0.2),
        [3, 1, 2],
        true
    );

    const [output] = lstm.forward(input);

    output.sum().backward();

    assert(input.grad != null, "Input gradient missing");
    assert(
        input.grad.data.some(v => Math.abs(v) > 1e-8),
        "Input gradient is zero"
    );
});

test("LSTM propagates gradients to initial hidden state", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 1 * 2).fill(0.2),
        [3, 1, 2]
    );

    const h0 = makeTensor(Array(3).fill(0.2), [1, 3], true);
    const c0 = makeTensor(Array(3).fill(0.3), [1, 3]);

    const [output] = lstm.forward(input, h0, c0);

    output.sum().backward();

    assert(h0.grad != null, "h0 gradient missing");
    assert(
        h0.grad.data.some(v => Math.abs(v) > 1e-8),
        "h0 gradient is zero"
    );
});

test("LSTM propagates gradients to initial cell state", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 1 * 2).fill(0.2),
        [3, 1, 2]
    );

    const h0 = makeTensor(Array(3).fill(0.2), [1, 3]);
    const c0 = makeTensor(Array(3).fill(0.3), [1, 3], true);

    const [output] = lstm.forward(input, h0, c0);

    output.sum().backward();

    assert(c0.grad != null, "c0 gradient missing");
    assert(
        c0.grad.data.some(v => Math.abs(v) > 1e-8),
        "c0 gradient is zero"
    );
});

test("LSTM gradients flow through all output timesteps", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(4 * 1 * 2).fill(0.2),
        [4, 1, 2],
        true
    );

    const [output] = lstm.forward(input);

    output.sum().backward();

    assert(input.grad != null, "Input gradient missing");

    for (let t = 0; t < 4; t++) {
        const start = t * 2;
        const end = start + 2;

        const hasGradient = input.grad.data
            .slice(start, end)
            .some(v => Math.abs(v) > 1e-8);

        assert(hasGradient, `No gradient at timestep ${t}`);
    }
});

test("LSTM gradients reach all parameters", () => {
    const lstm = new LSTM(2, 3);

    const input = makeTensor(
        Array(3 * 1 * 2).fill(0.2),
        [3, 1, 2],
        true
    );

    const [output, hn, cn] = lstm.forward(input);

    output.sum().add(hn.sum()).add(cn.sum()).backward();

    const parameters = lstm.parameters();

    assert(parameters.length > 0, "No parameters found");

    for (let i = 0; i < parameters.length; i++) {
        assert(
            parameters[i].grad != null,
            `Parameter ${i} gradient missing`
        );

        assert(
            allFinite(parameters[i].grad),
            `Parameter ${i} gradient is not finite`
        );
    }
});

// --------------------------------------------------
// 6. Numerical gradient checks
// --------------------------------------------------

test("LSTM input gradient numerical check", () => {
    const lstm = new LSTM(2, 2);

    const input = makeTensor(
        [0.2, 0.3, 0.4, 0.1],
        [2, 1, 2],
        true
    );

    const h0 = makeTensor([0.1, 0.2], [1, 2]);
    const c0 = makeTensor([0.3, 0.1], [1, 2]);

    const [output, hn, cn] = lstm.forward(input, h0, c0);

    output.sum().add(hn.sum()).add(cn.sum()).backward();

    for (let i = 0; i < input.data.length; i++) {
        checkNumericalGradient(
            input,
            i,
            input.grad.data[i],
            () => objective(lstm, input, h0, c0)
        );
    }
});

test("LSTM initial hidden gradient numerical check", () => {
    const lstm = new LSTM(2, 2);

    const input = makeTensor(
        [0.2, 0.3, 0.4, 0.1],
        [2, 1, 2]
    );

    const h0 = makeTensor([0.1, 0.2], [1, 2], true);
    const c0 = makeTensor([0.3, 0.1], [1, 2]);

    const [output, hn, cn] = lstm.forward(input, h0, c0);

    output.sum().add(hn.sum()).add(cn.sum()).backward();

    for (let i = 0; i < h0.data.length; i++) {
        checkNumericalGradient(
            h0,
            i,
            h0.grad.data[i],
            () => objective(lstm, input, h0, c0)
        );
    }
});

test("LSTM initial cell gradient numerical check", () => {
    const lstm = new LSTM(2, 2);

    const input = makeTensor(
        [0.2, 0.3, 0.4, 0.1],
        [2, 1, 2]
    );

    const h0 = makeTensor([0.1, 0.2], [1, 2]);
    const c0 = makeTensor([0.3, 0.1], [1, 2], true);

    const [output, hn, cn] = lstm.forward(input, h0, c0);

    output.sum().add(hn.sum()).add(cn.sum()).backward();

    for (let i = 0; i < c0.data.length; i++) {
        checkNumericalGradient(
            c0,
            i,
            c0.grad.data[i],
            () => objective(lstm, input, h0, c0)
        );
    }
});

test("LSTM recurrent parameter numerical gradient check", () => {
    const lstm = new LSTM(2, 2);

    const input = makeTensor(
        [0.2, 0.3, 0.4, 0.1],
        [2, 1, 2]
    );

    const h0 = makeTensor([0.1, 0.2], [1, 2]);
    const c0 = makeTensor([0.3, 0.1], [1, 2]);

    const [output, hn, cn] = lstm.forward(input, h0, c0);

    output.sum().add(hn.sum()).add(cn.sum()).backward();

    const parameters = lstm.parameters();

    // Check one element of every parameter tensor.
    for (let p = 0; p < parameters.length; p++) {
        const parameter = parameters[p];

        assert(parameter.grad != null, `Parameter ${p} gradient missing`);

        checkNumericalGradient(
            parameter,
            0,
            parameter.grad.data[0],
            () => objective(lstm, input, h0, c0),
            3e-2
        );
    }
});

// --------------------------------------------------
// 7. Configuration tests
// --------------------------------------------------

test("LSTM works without bias", () => {
    const lstm = new LSTM(2, 3, false);

    const input = makeTensor(
        Array(3 * 1 * 2).fill(0.2),
        [3, 1, 2]
    );

    const [output, hn, cn] = lstm.forward(input);

    assert(allFinite(output), "Invalid output");
    assert(allFinite(hn), "Invalid hidden state");
    assert(allFinite(cn), "Invalid cell state");
});

test("LSTM shares the same cell across timesteps", () => {
    const lstm = new LSTM(2, 3);

    const cell = lstm.cell;

    const input = makeTensor(
        Array(4 * 1 * 2).fill(0.2),
        [4, 1, 2]
    );

    lstm.forward(input);

    assert(
        lstm.cell === cell,
        "LSTM cell instance changed"
    );
});



// ==================================================
// SUMMARY
// ==================================================

log("\n================================");
log(`Results: ${passed} passed, ${failed} failed`);
log(`Total tests: ${passed + failed}`);
log("================================");

