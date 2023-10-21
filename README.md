# Active-Learning Machine Operated Molecular Dynamics (ALmoMD)
<br>
<div style="text-align:center">
	<img src="docs/logo.png" alt="ALmoMD logo" width="800"/>
</div>
<br>

[Active-learning machine-operated molecular dynamics (ALmoMD)](https://github.com/keysongkang/ALmoMD) is a Python code package designed for the effective training of machine learned interatomic potential (MLIP) through active learning based on uncertainty evaluation. It also facilitates the implementation of molecular dynamics (MD) using trained MLIPs with uncertainty evaluation.

The primary goal of ALmoMD is to efficiently train the MLIP without posing a challenge to _ab initio_ molecular dynamics, while effectively capturing rare dynamical events, leading to a more reliable MLIP. Additionally, ALmoMD includes the implementation of MD with uncertainty evaluation, providing an indication of its reliability.
<br>

- [Installation Guide](docs/installation.md)
- [Theoretical Background](docs/theory.md)
- [User Manuals](docs/documentation.md)
- [Tutorials](docs/tutorial.md)

## Code integration
ALmoMD utilizes multiple code packages to implement the active learning scheme for MLIP and MLIP-MD, with plans to introduce additional interfaces in the future.

- Atomic Simulation Environment ([ASE](https://wiki.fysik.dtu.dk/ase/))
- Machine Learned Interatomic Potential ([NequIP](https://github.com/mir-group/nequip))
- Density Functional Theory ([FHI-aims](https://fhi-aims.org/))
- *ab initio* Molecular dynamics and anharmonicity evalulation ([FHI-vibes](https://vibes-developers.gitlab.io/vibes/))


## License
MIT License

Copyright (c) [2023] [Kisung Kang]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.