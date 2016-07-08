# My Numerical MOOC

This is my work on the [**"Practical Numerical Methods with Python"**](http://openedx.seas.gwu.edu/courses/GW/MAE6286/2014_fall/about) MOOC by Lorena Barba.

I have worked through the resouces provided in this MOOC and its [GitHub repo](https://github.com/numerical-mooc/numerical-mooc). Which are divided in six different modules.

Here you can see my progress on this course given these are the **open bagdages** that I've earnt in this course. 

------------------
<img src="http://openedx.seas.gwu.edu:5000/fileearn.php?bgid=37131c110ab85abbb2b7698f87a71c5723ea20a4&7ccc956221a5da47d8e6304229fb49d71905bbb4" width="160">
------------------

### Course Modules

1. [**The phugoid model of glider flight.**](https://github.com/numerical-mooc/numerical-mooc/tree/master/lessons/01_phugoid)
Described by a set of two nonlinear ordinary differential equations, the phugoid model motivates numerical time integration methods, and we build it up starting from one simple equation, so that the unit can include 3 or 4 lessons  on initial value problems. This includes: a) Euler's method, 2nd-order RK, and leapfrog; b) consistency, convergence testing; c) stability
Computational techniques: array operations with NumPy; symbolic computing with SymPy; ODE integrators and libraries; writing and using functions.
2. [**Space and Time—Introduction to finite-difference solutions of PDEs.**](https://github.com/numerical-mooc/numerical-mooc/tree/master/lessons/02_spacetime)
Starting with the simplest model represented by a partial differential equation (PDE)—the linear convection equation in one dimension—, this module builds the foundation of using finite differencing in PDEs. (The module is based on the “CFD Python” collection, steps 1 through 4.)  It also motivates CFL condition, numerical diffusion, accuracy of finite-difference approximations via Taylor series, consistency and stability, and the physical idea of conservation laws.
Computational techniques: more array operations with NumPy and symbolic computing with SymPy; getting better performance with NumPy array operations.
3. [**Riding the wave: convection problems.**](https://github.com/numerical-mooc/numerical-mooc/tree/master/lessons/03_wave)
Starting with an overview of the concept of conservation laws, this module uses the traffic-flow model to study different solutions methods for problems with shocks: upwind, Lax-Friedrichs, Lax-Wendroff, MacCormack, then MUSCL (discussing limiters). Reinforces concepts of numerical diffusion and stability, in the context of solutions with shocks.  It will motivate spectral analysis of schemes, dispersion errors, Gibbs phenomenon, conservative schemes.
4. [**Spreading out: diffusion problems.**](https://github.com/numerical-mooc/numerical-mooc/tree/master/lessons/04_spreadout)
This module deals with solutions to parabolic PDEs, exemplified by the diffusion (heat) equation. Starting with the 1D heat equation, we learn the details of implementing boundary conditions and are introduced to implicit schemes for the first time. Another first in this module is the solution of a two-dimensional problem. The 2D heat equation is solved with both explicit and implict schemes, each time taking special care with boundary conditions. The final lesson builds solutions with a Crank-Nicolson scheme. 
5. [**Relax and hold steady: elliptic problems.**](https://github.com/numerical-mooc/numerical-mooc/tree/master/lessons/05_relax)
Laplace and Poisson equations (steps 9 and 10 of “CFD Python”), seen as systems relaxing under the influence of the boundary conditions and the Laplace operator. Iterative methods for algebraic equations resulting from discretizign PDEx: Jacobi method, Gauss-Seidel and successive over-relaxation methods. Conjugate gradient methods.
6. **Perform like a pro: making your codes run faster**
Getting performance out of your numerical Python codes with just-in-time compilation, targeting GPUs with Numba and PyCUDA.

Planned module (not yet started):
- **Boundaries take over: the boundary element method (BEM).**
Weak and boundary integral formulation of elliptic partial differential equations; the free space Green's function. Boundary discretization: basis functions; collocation and Galerkin systems. The BEM stiffness matrix: dense versus sparse;  matrix conditioning. Solving the BEM system: singular and near-singular integrals; Gauss quadrature integration.

### Thanks to
I am deeply thankful to [Lorena A. Barba](http://lorenabarba.com) (George Washington University, USA), [Ian Hawke](http://www.southampton.ac.uk/maths/about/staff/ih3.page) (Southampton University, UK), [Bernard Knaepen](http://depphys.ulb.ac.be/bknaepen/) (Université Libre de Bruxelles, Belgium) and all the other people who make posible this wonderful course.
