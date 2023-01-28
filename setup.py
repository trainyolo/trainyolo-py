import re
from pathlib import Path

from setuptools import setup

with open("README.md", "rb") as f:
    long_description = f.read().decode("utf-8")

with open(Path(__file__).parent / "trainyolo" / "__init__.py", "r") as f:
    content = f.read()
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

setup(
    name="trainyolo-py",
    version=version,
    author="trainyolo",
    author_email="info@trainyolo.com",
    description="Python sdk and cli for trainyolo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trainyolo/trainyolo-py",
    setup_requires=["wheel"],
    install_requires=[
        "tqdm",
        "requests",
        "argcomplete",
        "PyYAML"
    ],
    packages=[
        "trainyolo",
        "trainyolo.utils",
    ],
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
    entry_points={"console_scripts": ["trainyolo=trainyolo.cli:main"]},
    python_requires=">=3.6",
)