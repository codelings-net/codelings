# Codelings
*Digital organisms made out of computer code that live in the browser*

<Mick's blurb>

**TODO** Pointers to about.md, progress_so_far.md and debug/debug.md


## Installation

### Prerequisites

* **Python** - The main script ([`evolver.py`]) has been developed using Python 
3.8. It definitely needs at least version 3.6 (which introduced f-strings; 3.6 
or above is also required by the `wasmtime` module, which will be installed 
below). The script has not been tested with Python 3.6 or 3.7 (test reports 
welcome!).

* **The Python virtual environment module (`venv`)** - Recommended but not 
essential (if not using `venv`, simply skip any lines that have "env" or 
"activate" in them). The `venv` module sometimes comes as a separate package 
under Linux (e.g. `python3.8-venv`).

* **Git**

### Setting up a Python virtual environment

The following has been tested on Linux (bash) and Windows 10 (git-bash):

```bash
git clone https://github.com/codelings-net/codelings
cd codelings
python -m venv .env

# Linux bash
ln -s .env/bin/activate .

# Win10 git-bash
ln -s .env/Scripts/activate .

source activate
python -m pip install --upgrade pip
pip install wasmtime hexdump
```

(Reports from macOS and Windows PowerShell users would be welcome!)

The above should install the most recent version of the `wasmtime` WebAssembly 
runtime. The project has been developed using `wasmtime` version 0.25.0 and 
needs the fuel feature (which limits the number of instructions executed and 
thereby avoids infinite loops). Fuel was introduced in version 0.23.0 released 
in Feb 2021, so earlier versions will not work.

### Installing WABT

WABT (The WebAssembly Binary Toolkit, affectionately known as "wabbit") is not 
required to run the main script ([`evolver.py`]), but it is useful when 
experimenting with WebAssembly, e.g. for converting hand-written WebAssembly 
text files to binary (using `wat2wasm`) or for disassembling the binary files 
produced by the main script (via `wasm-objdump -d`).

**TODO**


## Usage

```bash
cd codelings
source activate

# print out a detailed helper message
./evolver.py -h

# print out a list of available scoring functions and their descriptions
./evolver.py -fn list

# generate 10 new generation 0 codelings of default length and print out
# their scores
./evolver.py -gen0 10 run01

# generate 10 strings of random bytes of length 5, score them with
# scoring function 'v02', print out their scores and save those with 
# score >= 0x00 to 'out'
./evolver.py -rnd0 10 -length 5 -fn v02 -thresh 0x00 run02
```

The default input directory is called `alive` and is assumed to contain 
codelings that are worth evolving further. The default output directory is 
`out`.

Most of the command-line options (including `-indir` and `-outdir`) can also be 
set in [`config.py`].

The script catches SIGINT and handles it gracefully, so if you want to end a 
run that's in progress, feel free to press Ctrl-C at any point. The script will 
finish jobs that have already started and then exit (usually within a few 
seconds).


## Useful links

### WebAssembly

Codelings are built out of WebAssembly instructions, so a passing 
familiarity with the basics is probably a good idea if you want to understand 
what's going on.

- Gentle intro: https://blog.scottlogic.com/2018/04/26/webassembly-by-hand.html

- The spec (1.1 draft): https://webassembly.github.io/spec/core/

- The spec (1.0, W3C): https://www.w3.org/TR/wasm-core-1/

- Index of instructions (1.1 draft): 
https://webassembly.github.io/spec/core/appendix/index-instructions.html

- Index of instructions (1.0, W3C):
https://www.w3.org/TR/wasm-core-1/#a7-index-of-instructions

- Main changes from 1.0 (aka the Minimum Viable Product or 'MVP') to 1.1:
https://github.com/WebAssembly/spec/tree/master/proposals

- Browser support for new features: https://webassembly.org/roadmap/

This project currently produces version 1.0 of WebAssembly. The goal is for all 
code to be executable in all 3 major browsers (Chrome, Firefox and Safari) with 
default settings. As of March 2021, this means that we can probably start using 
the multi-value feature from the 1.1 draft spec (functions being able to return 
multiple values; very useful) as well as import/export of mutable globals (less 
useful?).

### WABT

**TODO**

### Wasmtime-py

Wasmtime is a stand-alone WebAssembly runtime that allows WebAssembly programs 
to be run outside of the browser. This project uses the Python bindings.

- Source: https://github.com/bytecodealliance/wasmtime-py

- Examples: https://github.com/bytecodealliance/wasmtime-py/tree/main/examples

- Docs: https://bytecodealliance.github.io/wasmtime-py


[`config.py`]: config.py
[`evolver.py`]: evolver.py
