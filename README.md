# MusicNTD: Analyzing music with Nonnegative Tucker Decomposition #

Hello, and welcome on this framework!

This repository contains code used for the segmentation task of music using Nonnegative Tucker Decomposition (NTD).

You can download it using pip, by typing:

    pip install musicntd

Or download the source files and run the setup.py at the root of the folder.

Note that some Notebooks are included in the folder, along with their HTML (computed) version, in order to detail and present some experiments. By using pip, these Notebooks will be downloaded too, but may be hard to find if you're not used to search for the source files of the applications you install. In that sense, you should consider downloading the Notebooks folder manually. It will still work if you're using pip.

The code containing the NTD algorithm is not in this project but in another one, called nn-fac, also available on pip and at this address: https://gitlab.inria.fr/amarmore/nonnegative-factorization.

This project is still under development, and may contain bug. Comments are welcomed!

## Walkthrough Notebook ##

A walkthrough Notebook, to present how to use NTD for music segmentation, is available in the folder "Notebooks".

## Credits ##

Code was created by Axel Marmoret (<axel.marmoret@irisa.fr>), and strongly supported by Jeremy E. Cohen (<jeremy.cohen@irisa.fr>).

The technique in itself was also developed by Nancy Bertin (<nancy.bertin@irisa.fr>) and Frédéric Bimbot (<bimbot@irisa.fr>).
