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

nx = 64
ny = 64
mesh = UnitSquareMesh(nx, ny)

# TODO: Pick the spaces carefully
Vs = FunctionSpace(mesh, 'Lagrange', 1)
Vu = FunctionSpace(mesh, 'Lagrange', 2)


# TODO: Initialize these to the right values
# The true and inverted parameter
sigma_true = interpolate(Expression('log(2 + 7*(pow(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2),0.5) > 0.2))', degree=5),Vs)
sigma = interpolate(Expression("log(2.0)", degree=1),Vs)

mu_true = interpolate(Expression('log(2 + 7*(pow(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2),0.5) > 0.2))', degree=5),Vs)
mu = interpolate(Expression("log(2.0)", degree=1),Vs)

# define function for state and adjoint
u = Function(Vu)
v = Function(Vu)

# define Trial and Test Functions
u_trial, v_trial, sigma_trial = TrialFunction(Vu), TrialFunction(Vu), TrialFunction(Vs)
u_test, v_test, sigma_test = TestFunction(Vu), TestFunction(Vu), TestFunction(Vs)

# internal data
H = Constant("1.0")

# Gruneisen coefficient
Gamma = Constant("1.0")

# diffusion coefficient
gamma = Constant("1.0")

# true solution
u_true = Function(Vu)

# TODO: bc
# set up dirichlet boundary conditions
def boundary(x,on_boundary):
    return on_boundary

bc_state = DirichletBC(Vu, u0, boundary)
bc_adj = DirichletBC(Vu, Constant(0.), boundary)

#########################################################
# Setting up synthetic observations
#########################################################

noise_level = 0.05

F = inner(gamma * nabla_grad(u_trial), nabla_grad(u_test)) * dx + \
        sigma_true * u_trial * u_test * dx + \
        mu_true * abs(u_trial) * u_trial * u_test * dx

solve(F == 0, u_true, bcs = bc_state)


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

