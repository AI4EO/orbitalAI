# OrbitalAI - AI4EO Challenge

Welcome to the OrbitalAI AI4EO challenge !!

This repository contains the Jupyter notebooks, utility functions and utility data to get you started with simulating satellite imagery as is acquired on-board. You can simulate images to train your AI model for your at-the-edge application and win the challenge!

The challenge has two tracks:

 * _phisat-2_, where you will be developing an AI application using the multi-spectral imagery acquired on board of the Φ-sat-2 satellite;
 * _IMAGIN-e_, where you will be developing an AI application using the hyper-spectral imagery acquired on board of the International Space Station.

The two subfolders in this repo contain the notebooks to simulate the corresponding imagery, showing the simulation process, as well as describing how the pre-computed core datasets have been generated. 

## Installation

### Using pip

**Note:** Ensure you have installed GDAL-3.6.2 or install it trough anaconda as in the Dockerfile.

To install the required dependencies, execute:

```bash
pip3 install -r requirements.txt
```

### Using Docker

This project includes a Dockerfile and docker-compose.yml located in the .devcontainer directory. When using VSCode, you can utilize the Devcontainers extension to automatically manage extensions, including Jupyter, within your Docker container.

To launch the container, from the .devcontainer folder execute:

```bash
docker compose up 
```

## Help and support

For any issue relating to the notebooks or the data, open a ticket in the challenge Forum, so that we and the community can adequately support you. For improvements and fixes to the notebook, issues and pull requests can be opened in this AI4EO GitHub repository.
