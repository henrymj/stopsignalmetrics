import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stopsignalmetrics",
    version="0.0.0.5",
    author="Henry Jones",
    author_email="henrymj@stanford.edu",
    description="package for stop signal task metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henrymj/stopsignalmetrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
    install_requires=['numpy', 'pandas', 'scikit-learn']
)
