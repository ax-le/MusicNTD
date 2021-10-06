import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="musicntd",
    version="0.3.0",
    author="Marmoret Axel",
    author_email="axel.marmoret@irisa.fr",
    description="Package for NTD applied on musical segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/amarmore/musicntd",    
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3.7"
    ],
    license='BSD',
    install_requires=[
        'librosa >= 0.7.0',
        'madmom',
        'matplotlib',
        'mir_eval',
        'mirdata',
        'nn-fac == 0.2.0',
        'numpy >= 1.18.0',
        'pandas',
        'scipy >= 0.13.0',
        'soundfile',
        'tensorly >= 0.4.5',
    ],
    python_requires='>=3.7',
)