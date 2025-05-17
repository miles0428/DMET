## Module Physics Problem

### Class ProblemFomulation

#### class OneBodyFormulation
- attribute oneBodyHamiltonian
- function getAnalyticSolution -> slaterDeterminantMatrix
- function getRDM -> rdm 

#### class ManyBodyFormulation 
- attribute manyBodyHamiltonian 

## Module DMET

### class DMET
- attribute ProblemFormulation
- attribute fragments
- attribute problemSolver
- function getProjectors -> projectileMatrix
- function getFragmentHamiltonian -> fragmentHamiltonian 
- function getFragmentEnergy -> fragmentEnergy
- function solveFragment -> rdm
- function selfConsistent -> groundStateEnergy

## Module ProblemSolver

### class problemsolver
- function solve -> energy, rdm

