# **Autonomous navigation of anatomical structures**
A deep reinforcement learning-based simulation framework for autonomous robotic navigation of anatomical structures.

Note that the basis of the layout of this framework stems from the simulus ultrasound simulations repository (https://github.com/androst/simulus/).

## Installation

If you want to use the package in other projects, this can be achieved by installing the
package directly from the git repository or from source. Use these steps if you plan
to use the package without developing, running examples or tests.

**Option 1: Directly from repo using pip install**

First ensure that pip is up-to-date:
```bash
pip install --upgrade pip # Make sure pip is up to date first
```

Install the package directly from the github repository
```bash
pip install git+https://github.com/shavinr/autonomous_anatomic_navigation.git
```

**Option 2: From source**

You can also install it from source by downloading the repo from either github or bitbucket
```bash
git clone https://github.com/shavinr/autonomous_anatomic_navigation.git
cd simulus
pip install -e .
```

## Setup for development
Follow this step if you want to run examples, tests or contribute to the code.

**1. Clone repo**
```bash
git clone https://github.com/shavinr/autonomous_anatomic_navigation.git
```

**2. Setup virtual environment**

Ubuntu:
```bash
cd simulus
virtualenv --python=python3 venv
source venv/bin/activate
```

Windows:
Is unfortunately currently not supported.

A guide for packages with pip and virtual environments can be found
[here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

**3. Install requirements**
```bash
pip install --upgrade pip # Make sure pip is up to date first
pip install -r requirements.txt
pip install -e .
```

## Examples 
To visualise the CT simulation framework, run CT_simulator_env.py as main.
