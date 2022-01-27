# DECODER VARIABLE MISUSE TOOL

This project contains the last release of the code of the variable misuse tool deployment for the DECODER project. This tools allows to detect and repair bugs or errors corresponding to variables within source code files written in Java, C and C++.

More info about the project and its tools [here](https://www.decoder-project.eu/).


## First steps

The basic environment is created with the following command that only needs to be executed the first time the environment is setup:

```bash
conda env create -f environment.yml
```

Once the environment is created, it is preserved in between environment restarts, it can be activated as:

```bash
conda activate tensorflow-gpu
```

## Train and evaluate models

`Notebooks` foulder contains the training and the evaluation of the models for the Java and the C/C++ use cases. Training phase is carried out by the `hypeopt` library, in order to obtain the best model possible for both programming languages.

The result models are stored in the `models` foulder.

### Git LFS data

Be careful that the LFS files tracked are downloaded when you run the notebooks. If they are not downloaded, you can download all with:
```bash
git lfs pull
```
from the root directory of this repository.
