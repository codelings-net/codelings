# Codelings
"Digital organisms made out of computer code that live in the browser"


## Introduction

The goal of this project is to evolve digital organisms (called *codelings*) 
that will "live" in the browser and interact with users a bit like the 
[Tamagotchi](https://en.wikipedia.org/wiki/Tamagotchi) "digital pets" of the 
late 1990s - but with one key difference: unlike the Tamagotchis, and very much 
like biological organisms, codelings will be capable of evolving new features. 
In fact a key principle of the project is that every aspect of the organism can 
be mutated and evolved. If a user finds an interesting new mutation, they will 
be able to share it with others in the community. The hope is that as these 
mutations accumulate over time, we will see the emergence of complex digital 
creatures including some that might be able to recognise human faces, respond 
to simple voice commands etc.

The approach taken by this project is to start from scratch, i.e. from short 
strings of random bytes, and then attempt to engineer a positive feedback loop 
where a 'smart' mutator that has a better understanding of code than its 
ancestors is able to successfully evolve longer and more complex programs, 
including an improved version of itself, and so on until the 
[singularity](https://en.wikipedia.org/wiki/Technological_singularity).

Evolving the initial smart mutator by having humans examining millions of 
random programs one by one is clearly impractical, so instead the approach 
during early stages of the project will be to score programs on how well they 
can predict the next character in Wikipedia articles, and mutators will then 
compete on how well their mutations improve performance on this benchmark.

The programming language chosen for the project is 
[WebAssembly](https://webassembly.org/), because it is relatively simple, fast, 
cross-platform and available in all [major 
browsers](https://webassembly.org/roadmap/) without the user having to install 
any additional software. It is also a state-of-the-art sandbox for running 
untrusted code. The downside is that the type system and the control flow 
syntax make it significantly more brittle than typical assembly languages.


## Progress to date

#### Interface to WebAssembly

In order to keep the problem as simple as possible during early stages of the 
project, strings of WebAssembly code are embedded within an existing 
WebAssembly binary file that defines the following environment: **TODO**


#### Random sequences

**TODO**

|n_bytes | passed | attempts | success rate |
|:------:|-------:|---------:|:------------:|
|    1   |    128 |     256  |    50%       |
|    2   |  1.9e4 |   6.6e4  |    29%       |
|    3   |  1.4e5 |   1.0e6  |    14%       |
|    4   |  7.7e4 |   1.0e6  |     7.7%     |
|    5   |  1.2e4 |   1.0e6  |     1.2%     |
|    6   |  4.4e3 |   1.0e6  |     0.44%    |
|    7   |  2.2e3 |   1.0e6  |     0.22%    |
|    8   |  1.1e3 |   1.0e6  |     0.11%    |
|    9   |  5.6e2 |   1.0e6  |     0.056%   |
|   10   |  2.0e3 |   1.0e6  |     0.20%    |
|   11   |  1.3e3 |   1.0e6  |     0.13%    |
|   12   |  7.3e2 |   1.0e6  |     0.073%   |
|   13   |  3.7e2 |   1.0e6  |     0.037%   |
|   14   |  1.0e2 |   1.0e6  |     0.010%   |
|   15   |  0.4e2 |   1.0e6  |     0.004%   |
|   16   |  0.2e2 |   1.0e6  |     0.002%   |

####


## Useful links

#### WebAssembly

Codelings are built out of WebAssembly instructions and so a passing 
familiarity with the basics is probably a good idea if you want to 
understand what's going on.

- Gentle intro: https://blog.scottlogic.com/2018/04/26/webassembly-by-hand.html

- The spec: https://webassembly.github.io/spec/core/

- Index of instructions: 
https://webassembly.github.io/spec/core/appendix/index-instructions.html


#### WABT

**TODO**


#### Wasmtime-py

Wasmtime is a stand-alone WebAssembly runtime that allows WebAssembly programs 
to be run outside of the browser. This project uses the Python bindings.

- Source: https://github.com/bytecodealliance/wasmtime-py

- Examples: https://github.com/bytecodealliance/wasmtime-py/tree/main/examples

- Docs: https://bytecodealliance.github.io/wasmtime-py
