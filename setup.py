from setuptools import setup, find_packages

setup(
    name='ALmoMD',
    version='0.2.0',
    author='Kisung Kang',
    description='Active-learning machine-operated molecular dynamics (ALmoMD) is a Python code package designed for the effective training of machine learned interatomic potential (MLIP) through active learning based on uncertainty evaluation. It also facilitates the implementation of molecular dynamics (MD) using trained MLIPs with uncertainty evaluation.',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.0',
        'scipy==1.8.0',
        'son>=0.4.2',
        'ase>=3.22.1',
        'pandas>=1.5.3',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.2.2',
        'wandb>=0.13.10',
        'nequip>=0.5.6',
        'numba'
    ],
    entry_points={
        'console_scripts': [
            'almomd = almd.cli:main'
        ]
    },
)

