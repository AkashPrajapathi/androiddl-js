import { Linear, Module, ReLU } from "./androiddl.js";

export class ActorCritic extends Module {
    constructor() {
        super();

        this.shared1 = new Linear(11, 128, true, "he");
        this.shared2 = new Linear(128, 128, true, "he");

        this.actorHidden = new Linear(128, 64, true, "he");
        this.actor = new Linear(64, 3, true, "he");

        this.criticHidden = new Linear(128, 64, true, "he");
        this.critic = new Linear(64, 1, true, "he");

        this.relu = new ReLU();
    }

    forward(state) {
        let features = this.relu.call(
            this.shared1.call(state)
        );

        features = this.relu.call(
            this.shared2.call(features)
        );

        const actorFeatures = this.relu.call(
            this.actorHidden.call(features)
        );

        const criticFeatures = this.relu.call(
            this.criticHidden.call(features)
        );

        return {
            logits: this.actor.call(actorFeatures),
            value: this.critic.call(criticFeatures)
        };
    }
}