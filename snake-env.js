
export class SnakeEnv {
    constructor({
        mode = "player",
        canvas = null,
        gridSize = 20,
        cellSize = 20,
        stepInterval = 120,
        maxSteps = gridSize * gridSize * 2,
        renderOnStep = mode === "player"
    } = {}) {
        if (!["player", "ai"].includes(mode)) {
            throw new Error('mode must be "player" or "ai"');
        }

        this.mode = mode;
        this.canvas = canvas;
        this.gridSize = gridSize;
        this.cellSize = cellSize;
        this.stepInterval = stepInterval;
        this.maxSteps = maxSteps;

        this.renderOnStep = renderOnStep

        this.ctx = canvas?.getContext("2d") ?? null;

        if (canvas) {
            canvas.width = gridSize * cellSize;
            canvas.height = gridSize * cellSize;
        }

        this.timer = null;
        this.reset();

        if (this.mode === "player") {
            this.handleKeyDown = this.handleKeyDown.bind(this);
            window.addEventListener("keydown", this.handleKeyDown);
        }
    }

    // Directions: 0=up, 1=right, 2=down, 3=left
    get directionVector() {
        return [
            [0, -1],
            [1, 0],
            [0, 1],
            [-1, 0]
        ];
    }

    reset() {
        const center = Math.floor(this.gridSize / 2);

        this.snake = [
            { x: center, y: center },
            { x: center - 1, y: center },
            { x: center - 2, y: center }
        ];

        this.direction = 1;
        this.steps = 0;
        this.score = 0;
        this.done = false;

        this.food = this.spawnFood();

        if (this.renderOnStep) {
            this.draw();
        }

        return this.getState();
    }

    // Action: 0=straight, 1=turn right, 2=turn left
    step(action) {
        if (this.mode !== "ai") {
            throw new Error('step(action) is for mode: "ai"');
        }

        if (this.done) {
            throw new Error("Episode ended. Call reset() first.");
        }

        if (![0, 1, 2].includes(action)) {
            throw new Error("Action must be 0, 1, or 2");
        }

        return this.advance(action);
    }

    advance(action) {
        if (action === 1) {
            this.direction = (this.direction + 1) % 4;
        } else if (action === 2) {
            this.direction = (this.direction + 3) % 4;
        }

        const [dx, dy] = this.directionVector[this.direction];
        const head = this.snake[0];

        const next = {
            x: head.x + dx,
            y: head.y + dy
        };

        const ateFood =
            next.x === this.food.x &&
            next.y === this.food.y;

        const outside =
            next.x < 0 ||
            next.y < 0 ||
            next.x >= this.gridSize ||
            next.y >= this.gridSize;

        // Moving into the current tail cell is allowed if
        // the tail moves away on this step.
        const body = ateFood
            ? this.snake
            : this.snake.slice(0, -1);

        const hitBody = body.some(
            segment => segment.x === next.x && segment.y === next.y
        );

        this.steps++;

        let reward = -0.01;
        let terminated = false;
        let truncated = false;

        if (outside || hitBody) {
            reward = -1;
            terminated = true;
        } else {
            this.snake.unshift(next);

            if (ateFood) {
                reward = 1;
                this.score++;

                this.food = this.spawnFood();

                if (!this.food) {
                    // The board is completely filled.
                    reward = 1;
                    terminated = true;
                }
            } else {
                this.snake.pop();
            }

            if (!terminated && this.steps >= this.maxSteps) {
                truncated = true;
            }
        }

        this.done = terminated || truncated;

        const result = {
            state: this.getState(),
            reward,
            terminated,
            truncated,
            score: this.score
        };

        if (this.renderOnStep) {
            this.draw();
        }

        return result;
    }

    // 11 binary features:
    // danger ahead/right/left (3)
    // direction up/right/down/left (4)
    // food ahead/right/behind/left (4)
    getState() {
        const head = this.snake[0];

        const danger = [
            this.wouldCollide(this.direction),
            this.wouldCollide((this.direction + 1) % 4),
            this.wouldCollide((this.direction + 3) % 4)
        ];

        const direction = [0, 0, 0, 0];
        direction[this.direction] = 1;

        const foodDx = this.food ? this.food.x - head.x : 0;
        const foodDy = this.food ? this.food.y - head.y : 0;

        const [dx, dy] = this.directionVector[this.direction];

        // Project food direction onto the snake's heading
        // and its right-hand direction.
        const forward = foodDx * dx + foodDy * dy;
        const right = foodDx * dy - foodDy * dx;

        const foodDirection = [
            forward > 0 ? 1 : 0,
            right > 0 ? 1 : 0,
            forward < 0 ? 1 : 0,
            right < 0 ? 1 : 0
        ];

        return [...danger, ...direction, ...foodDirection];
    }

    wouldCollide(direction) {
        const [dx, dy] = this.directionVector[direction];
        const head = this.snake[0];

        const x = head.x + dx;
        const y = head.y + dy;

        if (
            x < 0 || y < 0 ||
            x >= this.gridSize || y >= this.gridSize
        ) {
            return 1;
        }

        // Tail vacates its cell when the snake does not eat.
        const body = this.snake.slice(0, -1);

        return body.some(
            segment => segment.x === x && segment.y === y
        ) ? 1 : 0;
    }

    spawnFood() {
        const occupied = new Set(
            this.snake.map(p => `${p.x},${p.y}`)
        );

        const available = [];

        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                if (!occupied.has(`${x},${y}`)) {
                    available.push({ x, y });
                }
            }
        }

        if (available.length === 0) return null;

        return available[
            Math.floor(Math.random() * available.length)
        ];
    }

    handleKeyDown(event) {
        if (this.mode !== "player" || this.done) return;

        const key = event.key.toLowerCase();

        const absoluteDirection = {
            arrowup: 0,
            w: 0,
            arrowright: 1,
            d: 1,
            arrowdown: 2,
            s: 2,
            arrowleft: 3,
            a: 3
        }[key];

        if (absoluteDirection === undefined) return;

        event.preventDefault();

        const difference =
            (absoluteDirection - this.direction + 4) % 4;

        // Ignore direct reversal.
        if (difference === 2) return;

        let action = 0;

        if (difference === 1) action = 1;
        if (difference === 3) action = 2;

        this.pendingAction = action;
    }

    start() {
        if (this.mode !== "player") {
            throw new Error("start() is for player mode");
        }

        this.stop();
        this.pendingAction = 0;

        this.timer = setInterval(() => {
            if (this.done) {
                this.stop();
                return;
            }

            const action = this.pendingAction ?? 0;
            this.pendingAction = 0;

            const result = this.advance(action);

            if (result.terminated || result.truncated) {
                this.stop();
            }
        }, this.stepInterval);
    }

    stop() {
        if (this.timer !== null) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }

    render() {
        this.draw();
    }

    draw() {
        if (!this.ctx) return;

        const { ctx, cellSize: size, gridSize } = this;

        ctx.fillStyle = "#111827";
        ctx.fillRect(0, 0, gridSize * size, gridSize * size);

        ctx.strokeStyle = "#1f2937";

        for (let i = 0; i <= gridSize; i++) {
            ctx.beginPath();
            ctx.moveTo(i * size, 0);
            ctx.lineTo(i * size, gridSize * size);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, i * size);
            ctx.lineTo(gridSize * size, i * size);
            ctx.stroke();
        }

        if (this.food) {
            ctx.fillStyle = "#ef4444";
            ctx.fillRect(
                this.food.x * size + 2,
                this.food.y * size + 2,
                size - 4,
                size - 4
            );
        }

        this.snake.forEach((segment, i) => {
            ctx.fillStyle = i === 0 ? "#a3e635" : "#4dba36";

            ctx.fillRect(
                segment.x * size + 1,
                segment.y * size + 1,
                size - 2,
                size - 2
            );
        });
    }

    dispose() {
        this.stop();

        if (this.handleKeyDown) {
            window.removeEventListener(
                "keydown",
                this.handleKeyDown
            );
        }
    }
}
