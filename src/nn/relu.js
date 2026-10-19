import { Module } from "./module.js";

export class ReLU extends Module {

    forward(x) {
        return x.relu()
    }

}