# IMAGIN-e track

Welcome to the IMAGIN-e track of the AI4EO challenge !!

This repository contains Jupyter notebooks, utility functions and the GeoDataFrames to get you started with simulating Hyper-Spectral Imagery and downloading the pre-computed datasets. We prepared two simulation workflows, starting from two different data collections, namely the Hyper-Spectral PRISMA imagery and the Multi-Spectral Sentinel-2 imagery.

## PRISMA

The folder `from-prisma` contains the code necessary to simulate HSI imagery available on board starting from PRISMA hyper-spectral imagery. Read through the `prisma2hsi-simulator.ipynb` notebook to understand the simulation workflow, and learn how you can download the pre-computed dataset. The pre-computed dataset has also been ingested as Bring Your Own Data collection into Sentinel Hub, meaning you can visualise it in EOBrowser. Instructions on how you can do so are in the `retrieve-hsi-from-sentinel-hub.ipynb` notebook. 

## Sentinel-2

The folder `from-sentinel2` contains the code necessary to simulate HSI imagery starting from Sentinel-2 imagery. Read through the `s22hsi-simulator.ipynb` notebook to understand the simulation workflow and learn how to directly download the pre-computed dataset.

## Requirements

Check the challenge webpage [https://platform.ai4eo.eu/orbitalai-imagin-e](https://platform.ai4eo.eu/orbitalai-imagin-e) for details on the goals of the challenge.

### Euro Data Cube

If you are running from the Euro Data Cube workspace, you are ready to go! Open the starter-pack notebook, follow the instructions in the notebook on how to retrieve the credential informations and get started.

### Your own resources

If you are using your own resources, consult either the `prisma2hsi-simulator.ipynb` or `s22hsi-simulator.ipynb` notebook to get familiar with the simulation workflow and to download the pre-computed imagery.

## Help and support

For any issue relating to the notebook or the data, open a ticket in the challenge Forum, so that we and the community can adequately support you. For improvements and fixes to the notebook, issues and pull requests can be opened in the AI4EO GitHub repository.
