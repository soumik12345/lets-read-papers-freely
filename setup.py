from setuptools import setup, find_packages

setup(
    name="research_paper_parser",
    version="0.1.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/soumik12345/lets-read-papers-freely",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.11",
    install_requires=open("requirements.txt").read().splitlines(),
)
