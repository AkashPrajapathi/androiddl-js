# AndroidDL.js

**A PyTorch-inspired deep learning library built from scratch in JavaScript.**

AndroidDL.js brings tensor computation, automatic differentiation, neural network modules, loss functions, probability distributions, and optimization to JavaScript.

Inspired by PyTorch, the project explores how deep learning frameworks work internally—from tensor operations and computation graphs to trainable parameters and recurrent neural network architectures.

- **Language:** JavaScript (ES Modules)
- **Package:** `@AkashPrajapathi/androiddl-js`
- **Package registry:** GitHub Packages
- **Build tool:** Vite

[GitHub Repository](https://github.com/AkashPrajapathi/androiddl-js) · [GitHub Package](https://github.com/AkashPrajapathi/androiddl-js/pkgs/npm/androiddl-js)

---

## Downloads

Download the latest release of AndroidDL.js from GitHub Releases.

**[Download AndroidDL.js Releases](https://github.com/AkashPrajapathi/androiddl-js/releases)**

You can download the prebuilt library files from the release assets or build the library from source using Vite.

- **Browser ES module:** `androiddl.js`
- **UMD build:** `androiddl.umd.cjs` (check browser compatibility before using it directly in a script tag)
- **Source code:** Available in this repository.

For npm installation, see the [Installation](#installation) section below.

---

## 🐍 Snake AI — Live Demo

See **AndroidDL.js in action** by running a trained Snake AI model directly in your browser.

👉 **[Play Snake AI Demo](https://akashprajapathi.github.io/androiddl-js/snake-inference.html)**

The demo loads trained model weights and runs inference using AndroidDL.js—no Python or server-side inference required.

### Features

- 🧠 Trained neural network inference in the browser
- ⚡ JavaScript-powered model execution
- 🎮 Play Snake with an AI agent
- 📦 Powered by AndroidDL.js

---

## Features

- **Tensor computation:** Multidimensional tensors with shape, stride, and storage-offset management.
- **Tensor operations:** Arithmetic, broadcasting, matrix multiplication, and reduction operations.
- **Tensor manipulation:** Slicing, concatenation, stacking, squeezing, unsqueezing, reshaping, and permutation.
- **Non-contiguous tensors:** Strided tensor views and operations on non-contiguous data.
- **Automatic differentiation:** Gradient tracking and reverse-mode automatic differentiation.
- **Neural networks:** Build custom models using modules, linear layers, and activation functions.
- **Recurrent neural networks:** RNN, GRU, and LSTM cells and sequence models.
- **Loss functions:** Mean squared error and cross-entropy loss.
- **Optimization:** Stochastic gradient descent with momentum support.
- **Probability distributions:** Categorical distribution for discrete sampling.
- **Modular architecture:** Compose reusable neural network components.
- **Browser support:** Use the generated JavaScript builds directly in browser applications.

---

## Installation

### Option 1: Install with npm

Install AndroidDL.js from GitHub Packages:

```bash
npm install @AkashPrajapathi/androiddl-js
```

If npm cannot locate the package, configure the GitHub Packages registry in your project's `.npmrc`:

```ini
@AkashPrajapathi:registry=https://npm.pkg.github.com
```

For a public package, try installing without a token first. If your environment requires authentication, follow the [GitHub Packages npm registry guide](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-npm-registry).

Import the library in JavaScript:

```javascript
import {
  Tensor,
  Module,
  Linear,
  ReLU,
  RNN,
  GRU,
  LSTM
} from "@AkashPrajapathi/androiddl-js";
```

### Option 2: Use directly in a browser

You can use the generated ES module build without installing the package through npm.

First, build the library:

```bash
npm install
npm run build
```

Copy the generated `dist/` directory into your web project, or serve it from your project's root.

Then import the library using a relative URL:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AndroidDL.js Example</title>
</head>
<body>
  <script type="module">
    import {
      Tensor,
      Linear
    } from "./dist/androiddl.js";

    const layer = new Linear(2, 3);

    const input = new Tensor([[1, 2]], [1, 2]);
    const output = layer.call(input);

    console.log(output);
  </script>
</body>
</html>
```

Serve the page through a local development server rather than opening the HTML file directly. This avoids browser module-loading restrictions.

**Build output:** The current Vite configuration generates `dist/androiddl.js` and `dist/androiddl.umd.cjs`. The examples above use the ES module build.

---

## Quick Start

### 1. Create tensors

```javascript
import { Tensor } from "@AkashPrajapathi/androiddl-js";

const a = new Tensor(
  [
    [1, 2],
    [3, 4]
  ],
  [2, 2]
);

const b = new Tensor(
  [
    [5, 6],
    [7, 8]
  ],
  [2, 2]
);

const sum = a.add(b);
const product = a.matmul(b);

console.log(sum);
console.log(product);
```

Tensors represent numerical data together with shape and stride information. Tensor operations can be combined to build larger computations.

### 2. Automatic differentiation

```javascript
import { Tensor } from "@AkashPrajapathi/androiddl-js";

const x = new Tensor([2], [], true);
const y = x.mul(x);

y.backward();

console.log(x.grad);
```

Here, \(y=x^2\). At \(x=2\), the derivative is:

\[
\frac{dy}{dx}=2x=4
\]

The third constructor argument enables gradient tracking. Calling `backward()` propagates gradients through the computation graph.

### 3. Build a neural network

```javascript
import {
  Tensor,
  Module,
  Linear,
  ReLU
} from "@AkashPrajapathi/androiddl-js";

class MyNetwork extends Module {
  constructor() {
    super();

    this.hidden = new Linear(2, 8);
    this.relu = new ReLU();
    this.output = new Linear(8, 1);
  }

  forward(input) {
    const hidden = this.relu.call(
      this.hidden.call(input)
    );

    return this.output.call(hidden);
  }
}

const model = new MyNetwork();

const input = new Tensor([[0.5, 1.0]], [1, 2]);
const prediction = model.call(input);

console.log(prediction);
```

This model accepts two input features, processes them through a hidden layer with eight neurons, and produces one output.

Extend `Module` and implement `forward()` to define your own models.

---

## Tensor API

`Tensor` is the foundation of AndroidDL.js. It provides numerical storage, multidimensional indexing, tensor operations, and automatic differentiation.

### Creating tensors

The Tensor constructor accepts data, shape, and an optional gradient-tracking argument.

```javascript
const x = new Tensor(
  [
    [1, 2, 3],
    [4, 5, 6]
  ],
  [2, 3],
  true
);
```

The third argument enables gradient tracking.

### Tensor properties

| Property | Description |
|---|---|
| `data` | Underlying numerical storage |
| `shape` | Dimensions of the tensor |
| `strides` | Memory steps used to access tensor elements |
| `offset` | Starting storage position of the tensor view |
| `numel` | Total number of elements |
| `grad` | Gradient accumulated during backpropagation |

### Arithmetic operations

| Method | Description |
|---|---|
| `add()` | Element-wise addition |
| `sub()` | Element-wise subtraction |
| `mul()` | Element-wise multiplication |
| `div()` | Element-wise division |
| `matmul()` | Matrix multiplication |

Arithmetic operations support tensor broadcasting where applicable.

Example:

```javascript
const a = new Tensor([1, 2, 3], [3]);
const b = new Tensor([10], [1]);

const result = a.add(b);

console.log(result);
```

### Tensor manipulation

| Method | Description |
|---|---|
| `reshape()` | Changes tensor shape |
| `permute()` | Reorders tensor dimensions |
| `transpose()` | Swaps two tensor dimensions |
| `slice()` | Selects ranges of elements along tensor dimensions |
| `squeeze()` | Removes dimensions of size one |
| `unsqueeze()` | Inserts dimensions of size one |

Example:

```javascript
const x = new Tensor(
  [
    [1, 2, 3],
    [4, 5, 6]
  ],
  [2, 3]
);

const transposed = x.transpose(0, 1);
const reshaped = x.reshape([3, 2]);
const expanded = x.unsqueeze(0);

console.log(transposed.shape);
console.log(reshaped.shape);
console.log(expanded.shape);
```

### Slicing

AndroidDL.js supports multidimensional slicing with positive and negative steps.

The slicing API supports:

- Selecting ranges using start and stop indices.
- Skipping elements using positive steps.
- Reversing dimensions using negative steps.
- Slicing multiple dimensions.
- Creating strided views over existing tensor storage.

Examples:

```javascript
const x = new Tensor(
  [0, 1, 2, 3, 4, 5],
  [6]
);

// Select indices 1, 2, and 3
const a = x.slice("1:4");

// Select every second element
const b = x.slice("0:6:2");

// Reverse the tensor
const c = x.slice("::-1");
```

Slicing is designed around tensor shape, strides, and storage offsets rather than requiring every selection to copy the underlying data.

### Reduction operations

| Method | Description |
|---|---|
| `sum()` | Computes the sum of tensor elements |
| `mean()` | Computes the mean |
| `min()` | Computes the minimum |
| `max()` | Computes the maximum |

Reduction operations can be used to calculate aggregate values and build loss functions.

### Mathematical operations

| Method | Description |
|---|---|
| `exp()` | Element-wise exponential |
| `log()` | Element-wise natural logarithm |
| `tanh()` | Hyperbolic tangent |

### Automatic differentiation methods

| Method | Description |
|---|---|
| `backward()` | Propagates gradients through the computation graph |
| `detach()` | Creates a tensor detached from gradient tracking |

AndroidDL.js uses reverse-mode automatic differentiation to calculate gradients through supported tensor operations.

### Tensor composition

AndroidDL.js supports combining multiple tensors.

| Operation | Description |
|---|---|
| `concatenate` | Joins tensors along an existing dimension |
| `stack` | Combines tensors along a new dimension |

Concatenation and stacking support multidimensional tensors and are designed to work with non-contiguous inputs.

---

## Core Neural Network Components

### Module

`Module` is the base class for neural network components. It provides a common interface for defining models and accessing their parameters.

```javascript
class MyModel extends Module {
  constructor() {
    super();

    this.layer = new Linear(4, 2);
  }

  forward(input) {
    return this.layer.call(input);
  }
}

const model = new MyModel();
const parameters = model.parameters();
```

Use `model.call(input)` to execute the forward pass and `model.parameters()` to retrieve the model's trainable parameters.

### Linear

`Linear` implements a fully connected neural network layer.

```javascript
const layer = new Linear(4, 2);

const input = new Tensor(
  [[1, 2, 3, 4]],
  [1, 4]
);

const output = layer.call(input);
```

The layer transforms four input features into two output features.

### Sequential

`Sequential` composes modules into a sequence.

```javascript
import {
  Sequential,
  Linear,
  ReLU
} from "@AkashPrajapathi/androiddl-js";

const model = new Sequential(
  new Linear(4, 8),
  new ReLU(),
  new Linear(8, 2)
);
```

### Activation functions

| Module | Purpose |
|---|---|
| `ReLU` | Rectified linear activation |
| `Sigmoid` | Sigmoid activation |
| `Tanh` | Hyperbolic tangent activation |
| `Softmax` | Softmax activation |

### Parameter

`Parameter` represents trainable model parameters.

Modules expose parameters through `parameters()`, allowing optimizers to update them during training.

---

## Recurrent Neural Networks

AndroidDL.js includes three major recurrent neural network architectures:

- Recurrent Neural Network (RNN)
- Gated Recurrent Unit (GRU)
- Long Short-Term Memory (LSTM)

These architectures process sequential data by maintaining hidden states across timesteps.

### RNN

A vanilla recurrent neural network processes a sequence one timestep at a time.

Available components:

- `RNNCell`
- `RNN`

The recurrent cell calculates a new hidden state using the current input and previous hidden state.

### GRU

The Gated Recurrent Unit uses update and reset gates to control information flow through the recurrent network.

Available components:

- `GRUCell`
- `GRU`

The GRU architecture provides gated state updates without maintaining a separate cell state.

### LSTM

The Long Short-Term Memory network uses a hidden state and a separate cell state to manage information across timesteps.

Available components:

- `LSTMCell`
- `LSTM`

The LSTM cell uses four gates:

- **Forget gate:** Controls which information is retained from the previous cell state.
- **Input gate:** Controls how much new information enters the cell state.
- **Candidate state:** Generates candidate information for the cell state.
- **Output gate:** Controls the information exposed through the hidden state.

### Working with sequences

The sequence models accept input tensors with the following layout:

```text
[sequence_length, batch_size, input_size]
```

For example, an input tensor with shape `[10, 4, 8]` represents:

- 10 timesteps
- 4 sequences in a batch
- 8 features per timestep

Each sequence model reuses its recurrent cell across timesteps.

### Sequence model outputs

The recurrent sequence models return their outputs and final states:

| Model | Forward result | Description |
|---|---|---|
| `RNN` | `[output, hn]` | Sequence output and final hidden state |
| `GRU` | `[output, hn]` | Sequence output and final hidden state |
| `LSTM` | `[output, hn, cn]` | Sequence output, final hidden state, and final cell state |

For these models:

- `output` contains the hidden state at every timestep.
- `hn` contains the final hidden state.
- `cn` contains the final cell state for LSTM.

The sequence output has shape:

```text
[sequence_length, batch_size, hidden_size]
```

The final hidden state has shape:

```text
[batch_size, hidden_size]
```

The LSTM final cell state has the same shape:

```text
[batch_size, hidden_size]
```

### Example: LSTM

```javascript
import {
  Tensor,
  LSTM
} from "@AkashPrajapathi/androiddl-js";

const sequenceLength = 10;
const batchSize = 4;
const inputSize = 8;
const hiddenSize = 16;

const model = new LSTM(inputSize, hiddenSize);

const input = new Tensor(
  new Float32Array(sequenceLength * batchSize * inputSize),
  [sequenceLength, batchSize, inputSize]
);

const [output, hn, cn] = model.call(input);

console.log(output.shape);
console.log(hn.shape);
console.log(cn.shape);
```

Expected shapes:

```text
output: [10, 4, 16]
hn:     [4, 16]
cn:     [4, 16]
```

RNN, GRU, and LSTM make AndroidDL.js suitable for experimenting with sequence modelling, time-series data, and recurrent neural network architectures.

---

## Training Components

### Loss functions

| Class | Purpose |
|---|---|
| `MSELoss` | Mean squared error |
| `CrossEntropyLoss` | Cross-entropy loss |

Example:

```javascript
import {
  Tensor,
  MSELoss
} from "@AkashPrajapathi/androiddl-js";

const prediction = new Tensor([0.8], [1]);
const target = new Tensor([1.0], [1]);

const lossFunction = new MSELoss();
const loss = lossFunction.call(prediction, target);

console.log(loss);
```

### SGD optimizer

`SGD` implements stochastic gradient descent and supports momentum and dampening options.

```javascript
import { SGD } from "@AkashPrajapathi/androiddl-js";

const optimizer = new SGD(model.parameters(), {
  lr: 0.01,
  momentum: 0.9,
  dampening: 0
});
```

A typical training workflow consists of:

1. Run the model's forward pass.
2. Calculate the loss.
3. Backpropagate gradients.
4. Update parameters with the optimizer.

Refer to the repository's examples for the precise optimizer lifecycle and gradient-reset API.

### Categorical distribution

`Categorical` provides a probability distribution for discrete outcomes.

It can be used in experiments involving probabilistic action selection and reinforcement learning.

---

## Public API

AndroidDL.js exposes the following components through its main entry point.

### Core

| Export | Description |
|---|---|
| `Tensor` | Multidimensional tensor computation and automatic differentiation |
| `Parameter` | Trainable neural network parameters |

### Neural Network Modules

| Export | Description |
|---|---|
| `Module` | Base class for neural network modules |
| `Linear` | Fully connected neural network layer |
| `Sequential` | Sequential composition of neural network modules |
| `ReLU` | Rectified linear activation |
| `Sigmoid` | Sigmoid activation |
| `Tanh` | Hyperbolic tangent activation |
| `Softmax` | Softmax activation |

### Recurrent Neural Networks

| Export | Description |
|---|---|
| `RNNCell` | Vanilla recurrent neural network cell |
| `RNN` | Sequence-based vanilla recurrent neural network |
| `GRUCell` | Gated Recurrent Unit cell |
| `GRU` | Sequence-based GRU network |
| `LSTMCell` | Long Short-Term Memory cell |
| `LSTM` | Sequence-based LSTM network |

### Loss Functions

| Export | Description |
|---|---|
| `MSELoss` | Mean squared error loss |
| `CrossEntropyLoss` | Cross-entropy loss |

### Optimization

| Export | Description |
|---|---|
| `SGD` | Stochastic gradient descent with momentum support |

### Probability Distributions

| Export | Description |
|---|---|
| `Categorical` | Categorical probability distribution |

### Importing the public API

All the above components can be imported from the package entry point:

```javascript
import {
  Tensor,
  Parameter,

  Module,
  Linear,
  Sequential,

  ReLU,
  Sigmoid,
  Tanh,
  Softmax,

  RNNCell,
  RNN,
  GRUCell,
  GRU,
  LSTMCell,
  LSTM,

  MSELoss,
  CrossEntropyLoss,

  SGD,
  Categorical
} from "@AkashPrajapathi/androiddl-js";
```

---

## Examples and Experiments

Explore the [GitHub repository](https://github.com/AkashPrajapathi/androiddl-js) for the implementation and available examples.

The project includes experiments involving:

- Tensor operations and automatic differentiation
- Tensor slicing and manipulation
- Neural network construction and training
- Recurrent neural networks: RNN, GRU, and LSTM
- Snake AI and reinforcement learning
- Neural network inference directly in the browser

---

## Project Goals

AndroidDL.js is an exploration of deep learning framework internals.

The project focuses on:

- Understanding tensor data structures and memory layout
- Implementing numerical operations and broadcasting
- Exploring computation graphs and automatic differentiation
- Designing a composable neural network module system
- Implementing recurrent neural network architectures
- Implementing loss functions and optimization algorithms
- Experimenting with neural networks in JavaScript applications
- Understanding how deep learning frameworks work internally by building one from scratch

---

## Contributing

Contributions, bug reports, feature requests, and suggestions are welcome.

1. Fork the repository.
2. Create a branch for your changes.
3. Implement and test your changes.
4. Submit a pull request describing the improvement.

For bugs and feature requests, use [GitHub Issues](https://github.com/AkashPrajapathi/androiddl-js/issues).

---

## License

AndroidDL.js is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

The MIT License permits use, modification, and distribution, including commercial use, subject to its terms.

---

**AndroidDL.js — Learn deep learning by building it, one operation at a time.**
