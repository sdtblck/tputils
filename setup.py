from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()
          
setup(
    name='tputils',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='Utilities for TPUs.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/sdtblck/tputils',
    author='Sid Black',
    author_email='sdtblck@gmail.com',
    install_requires=[
        'tpunicorn'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
