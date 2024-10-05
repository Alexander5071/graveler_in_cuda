# Graveler Fight Simulations in CUDA

This is a quick and (relatively) simple implementation of this [script](https://github.com/arhourigan/graveler/blob/main/graveler.py) made by Austin Hourigan of [ShoddyCast](https://www.youtube.com/@ShoddyCast), which aims to simulate the chances of escaping [this softlock](https://www.youtube.com/watch?v=GgMl4PrdQeo&t=0s) described by [Pikasprey Yellow](https://www.youtube.com/@Pikasprey). It is written in CUDA, as it is designed to run on Nvidia GPUs.

## Prerequisites
 * CUDA Toolkit - https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
 * make - https://gnuwin32.sourceforge.net/packages/make.htm

## Compilation
After installing the prerequisites and cloning this repo, simply running the `make` command in the root of the repo will compile the source code in the `graveler.exe` executable.

## Running
The executable can either be run as-is in a terminal or by using the command `make run`. The executable can receive a single argument, representing the number of "rolls" (matches) that the program should simulate. If no such argument is provided, the default (a billion/1.000.000.000) is used. This can also be specified when running through `make`, by running 

```shell
make run MAX_ROLLS=<number_of_rolls_desired>
```

## Implementation

The program uses CUDA threads to simulate multiple rolls at the same time and stops in the (extremely rare/almost impossible) event that a "winning" roll occured (which represents escaping the softlock).

To simulate the random possible rolls, the program uses the [cuRAND](https://docs.nvidia.com/cuda/curand/index.html) library, with a separate random state per thread, which are initialized once and used throughout the entire run.

## Results

A video showcasing some code runs can be found [here](https://youtu.be/ZxQZN3j9hsk). For the default value of one billion rolls, depending on how the program is timed (by the OS, which includes the full run, versus using the program's built-in timer, measuring the actual code being run), it finishes in approximately 2.2 seconds and 1.4 seconds, respectively. These times were obtained under Windows 10 on a laptop with an Nvidia GTX 1650.

## Final Thoughts

This is far from being very efficient/optimized and I am fully expecting there to be many better/faster submissions; however, this was a very fun way to spend an evening. Thanks Austin for launching this challenge, love your work and keep going <3
