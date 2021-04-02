from setuptools import setup, find_packages

setup(
    name='tputils',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='Utilities for TPUs.',
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
