# Makefile for the init_1d initial model generator

NDEBUG := 
MPI    :=
OMP    :=

COMP := gfortran

MKVERBOSE := t


# location of top-level MAESTRO directory
MAESTRO_TOP_DIR = ../../../

# EOS and network
EOS_DIR := helmeos
CONDUCTIVITY_DIR := timmes_stellar
NETWORK_DIR := general_null
GENERAL_NET_INPUTS := $(MICROPHYSICS_HOME)/networks/$(NETWORK_DIR)/aproxCNOsolar.net 



# the local sources needed to build the initial model generator
MODEL_SOURCES := init_1d.f90


include ../GInitModels.mak




