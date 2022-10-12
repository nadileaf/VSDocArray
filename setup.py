import os
import setuptools

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
with open(os.path.join(_cur_dir, "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mesoor_VSDocArray",
    version="1.0.0",
    author="Yusen Lin",
    author_email="yusenlin@mesoor.com",
    description="向量搜索服务，基于 Faiss",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nadileaf/VSDocArray",
    include_package_data=True,
    packages=setuptools.find_packages(),
    # package_data={
    #     # "vsss": ["schema/*.json"],
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=True,
)
