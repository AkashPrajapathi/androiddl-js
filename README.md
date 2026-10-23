# AndroidDL.js

**A PyTorch-inspired deep learning library built from scratch in JavaScript.**

AndroidDL.js brings tensor computation, automatic differentiation, neural network modules, loss functions, and optimization to JavaScript.

Inspired by PyTorch, the project explores how deep learning frameworks work internally—from tensor operations and computation graphs to trainable parameters and neural network training.

- **Language:** JavaScript (ES Modules)
- **Package:** `@AkashPrajapathi/androiddl-js`
- **Package registry:** GitHub Packages
- **Build tool:** Vite

[GitHub Repository](https://github.com/AkashPrajapathi/androiddl-js) · [GitHub Package](https://github.com/AkashPrajapathi/androiddl-js/pkgs/npm/androiddl-js)

## Downloads

Download the latest release of AndroidDL.js from GitHub Releases.

**[Download AndroidDL.js Releases](https://github.com/AkashPrajapathi/androiddl-js/releases)**

You can download the prebuilt library files from the release assets or build the library from source using Vite.

- **Browser ES module:** `androiddl.js`
- **UMD build:** `androiddl.umd.cjs` (check browser compatibility before using it directly in a script tag)
- **Source code:** Available in this repository.

For npm installation, see the [Installation](#installation) section below.

## 🐍 Snake AI — Live Demo

See **AndroidDL.js in action** by running a trained Snake AI model directly in your browser.

👉 **[Play Snake AI Demo](https://akashprajapathi.github.io/androiddl-js/snake-inference.html)**

The demo loads trained model weights and runs inference using AndroidDL.js—no Python or server-side inference required.

### Features
- 🧠 Trained neural network inference in the browser
- ⚡ JavaScript-powered model execution
- 🎮 Play Snake with an AI agent
- 📦 Powered by AndroidDL.js



## Features

- **Tensor computation:** Multidimensional tensors, shape and stride management, broadcasting, and matrix multiplication.
- **Automatic differentiation:** Gradient tracking and reverse-mode backpropagation.
- **Neural networks:** Build custom models using modules, linear layers, and activation functions.
- **Loss functions:** Mean squared error and cross-entropy loss.
- **Optimization:** Stochastic gradient descent with momentum support.
- **Modular architecture:** Compose reusable neural network components.
- **Browser support:** Use the generated JavaScript builds directly in browser applications.

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
  Linear,
  ReLU
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

### Option 3: Use a CDN

If you want to import AndroidDL.js from a CDN, use a CDN that supports GitHub Packages and can resolve the package's module entry point and dependencies.

For example, you can try:

```html
<script type="module">
  import {
    Tensor,
    Linear
  } from "https://esm.sh/@AkashPrajapathi/androiddl-js";

  const layer = new Linear(2, 3);
  console.log(layer);
</script>
```

CDN availability depends on package metadata and registry access. If the CDN cannot resolve the package, use the local browser build instead.

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

## Core Components

### Tensor

`Tensor` is the foundation of AndroidDL.js. It provides numerical storage and operations for multidimensional data.

Operations include:

- Element-wise arithmetic
- Broadcasting
- Matrix multiplication
- Reshape, permute, and transpose
- Reduction operations
- Gradient tracking and backpropagation

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

console.log(transposed.shape);
console.log(reshaped.shape);
```

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
const input = new Tensor([[1, 2, 3, 4]], [1, 4]);

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

| Module    | Purpose                       |
| --------- | ----------------------------- |
| `ReLU`    | Rectified linear activation   |
| `Sigmoid` | Sigmoid activation            |
| `Tanh`    | Hyperbolic tangent activation |
| `Softmax` | Softmax activation            |

### Parameter

`Parameter` represents trainable model parameters. Modules expose parameters through `parameters()`, allowing optimizers to update them during training.

## Training Components

### Loss functions

| Class              | Purpose            |
| ------------------ | ------------------ |
| `MSELoss`          | Mean squared error |
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

## Public API

The following components are exposed through the public entry point:

| Export             | Description                                     |
| ------------------ | ----------------------------------------------- |
| `Tensor`           | Numerical data, tensor operations, and autograd |
| `Parameter`        | Trainable parameters                            |
| `Module`           | Base class for neural network modules           |
| `Linear`           | Fully connected layer                           |
| `Sequential`       | Sequential module composition                   |
| `ReLU`             | ReLU activation                                 |
| `Sigmoid`          | Sigmoid activation                              |
| `Tanh`             | Tanh activation                                 |
| `Softmax`          | Softmax activation                              |
| `SGD`              | Stochastic gradient descent optimizer           |
| `MSELoss`          | Mean squared error loss                         |
| `CrossEntropyLoss` | Cross-entropy loss                              |

## Examples and Experiments

Explore the [GitHub repository](https://github.com/AkashPrajapathi/androiddl-js) for the implementation and available examples.

The project includes experiments involving:

- Tensor operations and automatic differentiation
- Neural network construction and training
- Snake AI and reinforcement learning

## Project Goals

AndroidDL.js is an exploration of deep learning framework internals.

The project focuses on:

- Understanding tensor data structures and memory layout
- Implementing numerical operations and broadcasting
- Exploring computation graphs and automatic differentiation
- Designing a composable neural network module system
- Implementing loss functions and optimization algorithms
- Experimenting with neural networks in JavaScript applications

## Contributing

Contributions, bug reports, feature requests, and suggestions are welcome.

1. Fork the repository.
2. Create a branch for your changes.
3. Implement and test your changes.
4. Submit a pull request describing the improvement.

For bugs and feature requests, use [GitHub Issues](https://github.com/AkashPrajapathi/androiddl-js/issues).

## License

AndroidDL.js is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

The MIT License permits use, modification, and distribution, including commercial use, subject to its terms.

---

**AndroidDL.js — Learn deep learning by building it, one operation at a time.**
