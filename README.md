# ExcelsiorBee
This is the backend for excelsior project. 
Please install and run this backend before running the [Excelsior App](https://github.com/subhasis-chand/excelsior)

The codes in this repository run on python3.
The following packages needs to be installed to run the programs.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.
install pip using this [link](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)

Make sure the pip works with python3 on your system. Otherwise install pip3 which binds to python3.

```bash
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install Flask
```

install [pytorch](https://pytorch.org/get-started/locally/).

## Start the server
Go inside the cloned repository. After installing all the dependancies, run

```bash
python api.py
```

This command assumes that 'python' command referes to python 3. Otherwise run
```bash
python3 api.py
```

Warning: This application does not work on python 2.