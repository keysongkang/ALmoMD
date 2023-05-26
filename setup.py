from setuptools import setup, find_packages

setup(
    name='almomd',
    version='1.0.0',
    author='Your Name',
    description='Your package description',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        # Add more dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            'almomd = almd.cli:main'
        ]
    },
)

