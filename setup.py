from setuptools import setup, find_packages

setup(
    name="arcgispro_ai",
    version="0.0.13",
    author="Danny McVey",
    author_email="dannybmcvey@gmail.com",
    description="AI tools for ArcGIS Pro",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/danmaps/arcgispro_ai",
    packages=find_packages(),
    install_requires=[
        "arcpy",  # Ensure arcpy is available in the runtime environment
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)