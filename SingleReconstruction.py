from dolfin import *
import numpy as np

import sys
sys.path.append( "../hippylib" )
from hippylib import *

import logging

import matplotlib.pyplot as plt
sys.path.append( "../hippylib/tutorial" )
import nb

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_active(False)

# TODO: Pick the spaces carefully
nx = 64
ny = 64
mesh = UnitSquareMesh(nx, ny)
Vs = FunctionSpace(mesh, 'Lagrange', 1)
Vu = FunctionSpace(mesh, 'Lagrange', 2)


# TODO: Initialize these to the right values
# The true and inverted parameter
sigma_true = interpolate(Expression('0.3 + 0.2 * sin(x[0]) * sin(x[1])', degree=5),Vs)
sigma = interpolate(Expression('0.3', degree=1),Vs)

mu_true = interpolate(Expression('log(2 + 7*(pow(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2),0.5) > 0.2))', degree=5),Vs)
mu = interpolate(Expression('0.3', degree=1),Vs)

# define function for state and adjoint
u = Function(Vu)
v = Function(Vu)

# define Trial and Test Functions
u_trial, v_trial, sigma_trial, mu_trial = TrialFunction(Vu), TrialFunction(Vu), TrialFunction(Vs), TrialFunction(Vs)
u_test, v_test, sigma_test, mu_test = TestFunction(Vu), TestFunction(Vu), TestFunction(Vs), TestFunction(Vs)


# internal data
H = Constant("1.0")

# Gruneisen coefficient
Gamma = Constant("1.0")

# diffusion coefficient
gamma = Expression('0.03 + 0.01 * sin(x[1])')

# true solution
u_true = Function(Vu)

# TODO: bc with g
# set up dirichlet boundary conditions
def boundary(x,on_boundary):
    return on_boundary

bc_state = DirichletBC(Vu, u0, boundary)
bc_adj = DirichletBC(Vu, Constant(0.), boundary)

#########################################################
# Setting up synthetic observations
#########################################################

noise_level = 0.05

F_true = inner(gamma * nabla_grad(u_trial), nabla_grad(u_test)) * dx + \
        sigma_true * u_trial * u_test * dx + \
        mu_true * abs(u_trial) * u_trial * u_test * dx

solve(F_true == 0, u_true, bcs = bc_state)


# TODO: Proper noise addition as per paper
H_data = project(Gamma * sigma_true * u_true + Gamma * mu_true * u_true * abs(u_true), Vu)
MAX = H_data.vector().norm("linf")
noise = Vector()


def mistmatch_cost(Gamma, sigma, mu, u, H):
    mismatch = Gamma * sigma * u + Gamma * mu * abs(u) * u - H
    return 0.5 * assemble(inner(mismatch, mismatch) * dx)

def regularization_cost(kappa, sigma, mu):
    regularization = assemble(inner(nabla_grad(sigma), nabla_grad(sigma))*dx) + \
            assemble(inner(nabla_grad(mu), nabla_grad(mu))*dx)
    return kappa * 0.5 * regularization

def L_BFGS(sigma, mu):
    '''
    Limited memory BFGS quasi-Newton method implementation used to 
    solve the minimization problem arising from the least squares reconstruction method.
    For more information about the limited-memory BFGS method, refer to 
    Numerical Optimization by Nocedal and Wright
    '''


    # TODO: Refactor gradient computation
    ###################################################################################
    # GRADIENT COMPUTATION
    ###################################################################################

    # semilinear forward equation to obtain u
    forward_eq = inner(gamma * nabla_grad(u_trial), nabla_grad(u_test)) * dx + \
        sigma * u_trial * u_test * dx + \
        mu * abs(u_trial) * u_trial * u_test * dx

    solve(forward_eq == 0, u, bcs = bc_state)

    z = Gamma * (sigma * u + mu * abs(u) * u) - H

    # linear adjoint equation to obtain v
    adjoint_eq_bilinear = inner(gamma * nabla_grad(v_trial), nabla_grad(v_test)) * dx + \
        (sigma + 2 * mu * abs(u)) * v_trial * v_test * dx

    adjoint_lhs = assemble(adjoint_eq_bilinear)

    adjoint_eq_L = -z * Gamma * (sigma + 2 * mu * abs(u)) * v_test * dx

    adjoint_rhs = assemble(adjoint_eq_L)

    solve(adjoint_lhs, v, adjoint_rhs)

    # use u and v obtain gradients of the objective function w.r.t. sigma and mu
    obj_d_sigma = z * Gamma * u * sigma_test * dx + v * u * sigma_test * dx  - \
        inner(kappa * nabla_grad(sigma), nabla_grad(sigma_test)) * dx

    obj_d_mu = z * Gamma * abs(u) * u * mu_test *  dx + v * abs(u) * u * mu_test * dx - \
        inner(kappa * nabla_grad(mu), nabla_grad(mu_test)) * dx    



