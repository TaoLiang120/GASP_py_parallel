# coding: utf-8
# Copyright (c) Henniggroup.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals, print_function

"""
Objects Maker module:

This module contains functions for creating the (singleton) objects
used by the genetic algorithm during the search.

"""

from gasp_p import general
from gasp_p import population
from gasp_p import geometry as geo
from gasp_p import variations
from gasp_p import energy_calculators
from gasp_p import organism_creators
from gasp_p import development

from pymatgen.core.structure import Structure

import os
import math

from mpi4py import MPI

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()


def make_objects(parameters, THIS_PATH):
    """
    Constructs the needed objects for the genetic algorithm search.

    Returns a dictionary containing the objects.

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file
    """

    # to hold all the objects
    objects_dict = {}

    # make the composition space object
    if 'CompositionSpace' in parameters:
        composition_space = general.CompositionSpace(
            parameters['CompositionSpace'])
    else:
        if rank_world == 0:
            print('Input file must contain a "CompositionSpace" block.')
            comm_world.Abort(rank_world)

    objects_dict['composition_space'] = composition_space

    # make the constraints object
    if 'Constraints' in parameters:
        if 'min_num_atoms' in parameters['Constraints']:
            if parameters['Constraints']['min_num_atoms'] < 2:
                if rank_world == 0:
                    print('The value passed to the "min_num_atoms" keyword in the '
                          'Constraints block must greater than or equal to 2.')
                    comm_world.Abort(rank_world)
        constraints = development.Constraints(parameters['Constraints'],
                                              composition_space)
    else:
        constraints = development.Constraints('default', composition_space)

    objects_dict['constraints'] = constraints

    # make the geometry object
    if 'Geometry' not in parameters:
        geometry = geo.Bulk()
    elif parameters['Geometry'] in (None, 'default'):
        geometry = geo.Bulk()
    elif 'shape' not in parameters['Geometry']:
        geometry = geo.Bulk()
    elif parameters['Geometry']['shape'] == 'cluster':
        geometry = geo.Cluster(parameters['Geometry'])
    elif parameters['Geometry']['shape'] == 'wire':
        geometry = geo.Wire(parameters['Geometry'])
    elif parameters['Geometry']['shape'] == 'sheet':
        geometry = geo.Sheet(parameters['Geometry'])
    # TODO: add any other non-bulk geometries here
    else:
        geometry = geo.Bulk()

    objects_dict['geometry'] = geometry

    # make the development object
    if 'Development' in parameters:
        developer = development.Developer(parameters['Development'], geometry)
    else:
        developer = development.Developer('default', geometry)

    objects_dict['developer'] = developer

    # make the redundancy guard object
    if 'RedundancyGuard' in parameters:
        redundancy_guard = development.RedundancyGuard(
            parameters['RedundancyGuard'], geometry)
    else:
        redundancy_guard = development.RedundancyGuard('default', geometry)

    objects_dict['redundancy_guard'] = redundancy_guard

    # make the id generator
    id_generator = general.IDGenerator()
    objects_dict['id_generator'] = id_generator

    # make the organism creators
    initial_organism_creators = make_organism_creators(
        parameters, composition_space, constraints)

    # if more than one organism creator, sort them so that the attempts-based
    # ones are at the front and the successes-based ones are at the back
    if len(initial_organism_creators) > 1:
        initial_organism_creators.sort(key=lambda x: x.is_successes_based)

    objects_dict['organism_creators'] = initial_organism_creators

    if 'NumCalcsAtOnce' not in parameters:
        num_calcs_at_once = 1
    elif parameters['NumCalcsAtOnce'] in (None, 'default'):
        num_calcs_at_once = 1
    else:
        num_calcs_at_once = parameters['NumCalcsAtOnce']

    objects_dict['num_calcs_at_once'] = num_calcs_at_once

    if 'NumProcTask' not in parameters:
        nproc_task = 1
    elif parameters['NumProcTask'] in (None, 'default'):
        nproc_task = 1
    else:
        nproc_task = parameters['NumProcTask']

    objects_dict['nproc_task'] = nproc_task

    if 'NumCalcsRestartFile' not in parameters:
        num_calcs_to_restartfile = 50
    elif parameters['NumCalcsRestartFile'] in (None, 'default'):
        num_calcs_to_restartfile = 50
    else:
        num_calcs_to_restartfile = parameters['NumCalcsRestartFile']

    objects_dict['num_calcs_to_restartfile'] = num_calcs_to_restartfile

    # get the run title
    if 'RunTitle' not in parameters:
        run_dir_name = 'garun'
    elif parameters['RunTitle'] in (None, 'default'):
        run_dir_name = 'garun'
    else:
        run_dir_name = 'garun_' + str(parameters['RunTitle'])
    objects_dict['run_dir_name'] = run_dir_name

    if 'Screen' not in parameters:
        Screen = True
    elif parameters['Screen'] is None:
        Screen = False
    elif isinstance(parameters['Screen'], bool):
        Screen = parameters['Screen']
    else:
        Screen = True
    objects_dict['Screen'] = Screen

    if 'Log' not in parameters:
        Log = True
    elif parameters['Log'] is None:
        Log = False
    elif isinstance(parameters['Log'], bool):
        Log = parameters['Log']
    else:
        Log = True
    objects_dict['Log'] = Log

    # make the energy calculator
    energy_calculator = make_energy_calculator(parameters, geometry,
                                               composition_space, THIS_PATH)
    objects_dict['energy_calculator'] = energy_calculator

    # make the stopping criteria
    stopping_criteria = make_stopping_criteria(parameters, composition_space)
    objects_dict['stopping_criteria'] = stopping_criteria

    # determine which variations should have non-zero default fractions
    do_permutation = False
    if len(composition_space.get_all_swappable_pairs()) > 0:
        do_permutation = True

    # get the number of atoms per composition
    if len(composition_space.endpoints) > 1:
        atoms_per_comp = len(composition_space.endpoints)
    else:
        atoms_per_comp = \
            composition_space.endpoints[0].reduced_composition.num_atoms

    # see if numstoichmut can be done w/o violating the min or max number of
    # atoms constraints
    do_atomsmut = False
    if composition_space.objective_function == 'pd':
        if constraints.min_num_atoms != constraints.max_num_atoms:
            do_atomsmut = True
    elif composition_space.objective_function == 'epa':
        bottom = int(math.ceil(constraints.min_num_atoms / atoms_per_comp))
        top = int(math.floor(constraints.max_num_atoms / atoms_per_comp))
        if top > bottom:
            do_atomsmut = True

    # set default fractions for the variations
    default_variation_fractions = {}
    if do_permutation and do_atomsmut:
        default_variation_fractions['permutation'] = 0.1
        default_variation_fractions['num_atoms_mut'] = 0.1
        default_variation_fractions['structure_mut'] = 0.1
        default_variation_fractions['mating'] = 0.7
    elif not do_permutation and do_atomsmut:
        default_variation_fractions['permutation'] = 0.0
        default_variation_fractions['num_atoms_mut'] = 0.1
        default_variation_fractions['structure_mut'] = 0.1
        default_variation_fractions['mating'] = 0.8
    elif do_permutation and not do_atomsmut:
        default_variation_fractions['permutation'] = 0.1
        default_variation_fractions['num_atoms_mut'] = 0.0
        default_variation_fractions['structure_mut'] = 0.1
        default_variation_fractions['mating'] = 0.8
    elif not do_permutation and not do_atomsmut:
        default_variation_fractions['permutation'] = 0.0
        default_variation_fractions['num_atoms_mut'] = 0.0
        default_variation_fractions['structure_mut'] = 0.2
        default_variation_fractions['mating'] = 0.8

    # make the variations
    variations_list = make_variations(parameters, default_variation_fractions,
                                      composition_space)

    # check that at least one variation has been used
    if len(variations_list) == 0:
        if rank_world == 0:
            print('At least one variation must be used. Either leave entire '
                  '"Variations" block blank to use default variations, or specify '
                  'at least one variation within the "Variations" block.')
            comm_world.Abort(rank_world)

    # check that the variations' fraction variables sum to 1
    frac_sum = 0.0
    for variation in variations_list:
        frac_sum = frac_sum + variation.fraction
    if frac_sum < 0.999 or frac_sum > 1.001:
        if rank_world == 0:
            print("The Variations' fraction values must sum to 1.")
            comm_world.Abort(rank_world)

    objects_dict['variations'] = variations_list

    # make the pool, selection, and composition fitness weight
    if 'Pool' not in parameters:
        pool = population.Pool(None, composition_space, run_dir_name)
    else:
        if 'num_promoted' in parameters['Pool']:
            if parameters['Pool']['num_promoted'] < 1:
                if rank_world == 0:
                    print('At least one organism must be promoted in the Pool.')
                    comm_world.Abort(rank_world)
            else:
                pool = population.Pool(parameters['Pool'], composition_space,
                                       run_dir_name)
        else:
            pool = population.Pool(parameters['Pool'], composition_space,
                                   run_dir_name)

    if 'Selection' not in parameters:
        selection = general.SelectionProbDist(None, pool.size)
    else:
        selection = general.SelectionProbDist(parameters['Selection'],
                                              pool.size)

    if 'CompositionFitnessWeight' not in parameters:
        comp_fitness_weight = general.CompositionFitnessWeight(None)
    else:
        if 'max_weight' in parameters['CompositionFitnessWeight']:
            if parameters['CompositionFitnessWeight']['max_weight'] < 0 or \
                    parameters['CompositionFitnessWeight']['max_weight'] > 1:
                if rank_world == 0:
                    print('The maximum weight of the composition fitness must lie'
                          ' in the interval [0,1].')
                    print('Please change the value passed to the "max_weight" '
                          'keyword in the CompositionFitnessWeight block.')
                    comm_world.Abort(rank_world)
            else:
                comp_fitness_weight = general.CompositionFitnessWeight(
                    parameters['CompositionFitnessWeight'])
        else:
            comp_fitness_weight = general.CompositionFitnessWeight(
                parameters['CompositionFitnessWeight'])

    pool.selection = selection
    pool.comp_fitness_weight = comp_fitness_weight
    objects_dict['pool'] = pool

    return objects_dict


def make_organism_creators(parameters, composition_space, constraints):
    """
    Returns a list containing organism creator objects.

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file

        composition_space: the CompositionSpace of the search

        constraints: the Constraints of the search
    """

    if 'InitialPopulation' not in parameters:
        return make_default_organism_creator(composition_space, constraints)
    elif parameters['InitialPopulation'] in (None, 'default'):
        return make_default_organism_creator(composition_space, constraints)
    # make the specified creators
    else:
        # check that at least one valid option is used
        # TODO: if other organism creators are used, check them here as well
        if 'random' not in parameters['InitialPopulation'] and 'from_files' \
                not in parameters['InitialPopulation']:
            if rank_world == 0:
                print('At least one valid option for making structures for the '
                      'initial population must be provided.')
                print('Please use the "random" and/or "from_files" keywords in '
                      'the InitialPopulation block.')
                comm_world.Abort(rank_world)

        initial_organism_creators = []

        # the random organism creator
        if 'random' in parameters['InitialPopulation']:
            random_organism_creator = organism_creators.RandomOrganismCreator(
                parameters['InitialPopulation']['random'], composition_space,
                constraints)
            initial_organism_creators.append(random_organism_creator)

        # the from files organism creator
        if 'from_files' not in parameters['InitialPopulation']:
            if composition_space.objective_function == 'pd':
                if rank_world == 0:
                    print('For phase diagram searches, reference structures at '
                          'each endpoint of the composition space must be '
                          'provided.')
                    print('Please use the "from_files" keyword in the '
                          'InitialPopulation block to provide the reference '
                          'structures.')
                    comm_world.Abort(rank_world)
        # if nothing is given after the from_files keyword
        elif parameters['InitialPopulation']['from_files'] is None:
            if rank_world == 0:
                print('The path to the folder containing the files must be '
                      'provided. Please use the "path_to_folder" keyword.')
                comm_world.Abort(rank_world)
        # if path_to_folder keyword is not given
        elif 'path_to_folder' not in parameters['InitialPopulation'][
            'from_files']:
            if rank_world == 0:
                print('Incorrect keyword given after "from_files" in the '
                      'InitialPopulation block. Please use the "path_to_folder" '
                      'keyword.')
                comm_world.Abort(rank_world)
        else:
            given_path = parameters['InitialPopulation']['from_files'][
                'path_to_folder']
            # if no path was given after path_to_folder keyword
            if given_path is None:
                if rank_world == 0:
                    print('The path to the folder containing the files for the '
                          'initial population must be provided. Please give the '
                          'path after the "path_to_folder" keyword.')
                    comm_world.Abort(rank_world)
            # if the given path does not exist
            elif not os.path.exists(given_path):
                if rank_world == 0:
                    print('The given folder containing structures for the initial '
                          'population does not exist.')
                    comm_world.Abort(rank_world)
            # if the folder exists, check that it contains files
            elif len([f for f in os.listdir(given_path) if
                      os.path.isfile(os.path.join(given_path, f))]) == 0:
                if rank_world == 0:
                    print('The given folder containing structures for the initial '
                          'population does not contain any files.')
                    comm_world.Abort(rank_world)
            else:
                files_organism_creator = organism_creators.FileOrganismCreator(
                    given_path)
                # check that the files cover all composition space endpoints
                if composition_space.objective_function == 'pd':
                    cells = files_organism_creator.get_cells()
                    provided_endpoints = []
                    for endpoint in composition_space.endpoints:
                        for cell in cells:
                            if cell.composition.reduced_composition.almost_equals(
                                    endpoint.reduced_composition) and \
                                    endpoint not in provided_endpoints:
                                provided_endpoints.append(endpoint)
                    # check if we got them all
                    for endpoint in composition_space.endpoints:
                        if endpoint not in provided_endpoints:
                            if rank_world == 0:
                                print('Error: valid structure files not provided '
                                      'to the initial population for all '
                                      'endpoints of the composition space.')
                                comm_world.Abort(rank_world)
                initial_organism_creators.append(files_organism_creator)

        # TODO: if other organism creators are used, they should be
        # instantiated here

        return initial_organism_creators


def make_default_organism_creator(composition_space, constraints):
    """
    Returns a list containing a RandomOrganismCreator, or quits.

    Args:
        composition_space: the CompositionSpace of the search

        constraints: the Constraints of the search
    """

    if composition_space.objective_function == 'pd':
        if rank_world == 0:
            print('For phase diagram searches, reference structures at each '
                  'endpoint of the composition space must be provided in the '
                  'initial population.')
            print('Please use the "from_files" keyword in the '
                  'InitialPopulation block to provide the reference '
                  'structures.')
            comm_world.Abort(rank_world)
    else:
        random_organism_creator = organism_creators.RandomOrganismCreator(
            'default', composition_space, constraints)
        return [random_organism_creator]


def make_energy_calculator(parameters, geometry, composition_space, THIS_PATH):
    """
    Returns an EnergyCode object corresponding to which energy code was
    specified in the input file. Quits if an energy code object cannot be made.

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file

        geometry: the Geometry for the search

        composition_space: the CompositionSpace of the search
    """

    if 'EnergyCode' not in parameters:
        if rank_world == 0:
            print('A method for calculating energy must be provided. Please use '
                  'the "EnergyCode" keyword.')
            comm_world.Abort(rank_world)
    elif parameters['EnergyCode'] is None:
        if rank_world == 0:
            print('An energy code must be specified after the "EnergyCode" '
                  'keyword.')
            comm_world.Abort(rank_world)
    # for GULP
    elif 'gulp' in parameters['EnergyCode']:
        return make_gulp_energy_calculator(parameters, geometry, THIS_PATH)
    # for LAMMPS
    elif 'pylammps' in parameters['EnergyCode']:
        return make_pylammps_energy_calculator(parameters, geometry, THIS_PATH)
    # for pylammps
    elif 'lammps' in parameters['EnergyCode']:
        return make_lammps_energy_calculator(parameters, geometry, THIS_PATH)
    # for VASP
    elif 'vasp' in parameters['EnergyCode']:
        return make_vasp_energy_calculator(parameters, composition_space,
                                           geometry, THIS_PATH)
    else:
        if rank_world == 0:
            print('The given energy code name is invalid.')
            comm_world.Abort(rank_world)


def make_gulp_energy_calculator(parameters, geometry, THIS_PATH):
    """
    Returns a GulpEnergyCalculator object, or quits if one cannot be made.

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file

        geometry: the Geometry for the search
    """

    if parameters['EnergyCode']['gulp'] is None:
        if rank_world == 0:
            print('No GULP header or potential files given. Please use the '
                  '"header_file" and "potential_file" keywords.')
            comm_world.Abort(rank_world)
    else:
        # get the header file
        if 'header_file' not in parameters['EnergyCode']['gulp']:
            if rank_world == 0:
                print('A GULP header file must be provided. Please use the '
                      '"header_file" keyword.')
                comm_world.Abort(rank_world)
        elif parameters['EnergyCode']['gulp']['header_file'] is None:
            if rank_world == 0:
                print('No GULP header file given after the "header_file" '
                      'keyword. Please provide one.')
                comm_world.Abort(rank_world)
        else:
            # get the path to the header file
            header_file_path = parameters['EnergyCode']['gulp'][
                'header_file']
            # check that the header file exists
            if not os.path.exists(header_file_path):
                if rank_world == 0:
                    print('The given GULP header file does not exist.')
                    comm_world.Abort(rank_world)
        # get the potential file
        if 'potential_file' not in parameters['EnergyCode']['gulp']:
            if rank_world == 0:
                print('A GULP potential file must be provided. Please use the '
                      '"potential_file" keyword.')
                comm_world.Abort(rank_world)
        elif parameters['EnergyCode']['gulp']['potential_file'] is None:
            if rank_world == 0:
                print('No GULP potential file given after the '
                      '"potential_file" keyword. Please provide one.')
                comm_world.Abort(rank_world)
        else:
            # get the path to the potential file
            potential_file_path = parameters['EnergyCode']['gulp'][
                'potential_file']
            # check that the potential file exists
            if not os.path.exists(potential_file_path):
                if rank_world == 0:
                    print('The given GULP potential file does not exist.')
                    comm_world.Abort(rank_world)

        if 'path_to_callscript' not in parameters['EnergyCode']['gulp']:
            path_to_callscript = THIS_PATH
        elif parameters['EnergyCode']['gulp']['path_to_callscript'] in (None, 'default'):
            path_to_callscript = THIS_PATH
        else:
            path_to_callscript = parameters['EnergyCode']['gulp']['path_to_callscript']
        if 'callscript' not in parameters['EnergyCode']['gulp']:
            callscript = "callgulp"
        elif parameters['EnergyCode']['gulp']['callscript'] in (None, 'default'):
            callscript = "callgulp"
        else:
            callscript = parameters['EnergyCode']['gulp']['callscript']

        return energy_calculators.GulpEnergyCalculator(
            header_file_path, potential_file_path, geometry, path_to_callscript=path_to_callscript,
            callscript=callscript)


def make_lammps_energy_calculator(parameters, geometry, THIS_PATH):
    """
    Returns a LammpsEnergyCalculator object, or quits if one cannot be made.

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file

        geometry: the Geometry for the search
    """

    if parameters['EnergyCode']['lammps'] is None:
        if rank_world == 0:
            print('No LAMMPS input script given. Please use the "input_script" '
                  'keyword.')
            comm_world.Abort(rank_world)
    else:
        # get the input script
        if 'input_script' not in parameters['EnergyCode']['lammps']:
            if rank_world == 0:
                print('A LAMMPS input script must be provided. Please use the '
                      '"header_file" keyword.')
                comm_world.Abort(rank_world)
        elif parameters['EnergyCode']['lammps']['input_script'] is None:
            if rank_world == 0:
                print('No LAMMPS input script given after the "input_script" '
                      'keyword. Please provide one.')
                comm_world.Abort(rank_world)
        else:
            # get the path to the input script
            input_script_path = parameters['EnergyCode']['lammps'][
                'input_script']
            # check that the input script exists
            if not os.path.exists(input_script_path):
                if rank_world == 0:
                    print('The given LAMMPS input script does not exist.')
                    comm_world.Abort(rank_world)

        if 'path_to_callscript' not in parameters['EnergyCode']['lammps']:
            path_to_callscript = THIS_PATH
        elif parameters['EnergyCode']['lammps']['path_to_callscript'] in (None, 'default'):
            path_to_callscript = THIS_PATH
        else:
            path_to_callscript = parameters['EnergyCode']['lammps']['path_to_callscript']
        if 'callscript' not in parameters['EnergyCode']['lammps']:
            callscript = "calllammps"
        elif parameters['EnergyCode']['lammps']['callscript'] in (None, 'default'):
            callscript = "calllammps"
        else:
            callscript = parameters['EnergyCode']['lammps']['callscript']
        return energy_calculators.LammpsEnergyCalculator(
            input_script_path, geometry, path_to_callscript=path_to_callscript, callscript=callscript)


def make_pylammps_energy_calculator(parameters, geometry, THIS_PATH):
    """
    Returns a PyLammpsEnergyCalculator object, or quits if one cannot be made.

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file

        geometry: the Geometry for the search
    """

    if parameters['EnergyCode']['pylammps'] is None:
        if rank_world == 0:
            print('No LAMMPS input script given. Please use the "input_script" '
                  'keyword.')
            comm_world.Abort(rank_world)
    else:
        # get the input script
        if 'input_script' not in parameters['EnergyCode']['pylammps']:
            if rank_world == 0:
                print('A PYLAMMPS input script must be provided.')
                comm_world.Abort(rank_world)
        elif parameters['EnergyCode']['pylammps']['input_script'] is None:
            if rank_world == 0:
                print('No LAMMPS input script given after the "input_script" '
                      'keyword. Please provide one.')
                comm_world.Abort(rank_world)
        else:
            # get the path to the input script
            input_script_path = parameters['EnergyCode']['pylammps'][
                'input_script']
            # check that the input script exists
            if not os.path.exists(input_script_path):
                if rank_world == 0:
                    print('The given LAMMPS input script does not exist.')
                    comm_world.Abort(rank_world)

        return energy_calculators.PyLammpsEnergyCalculator(
            input_script_path, geometry)


def make_vasp_energy_calculator(parameters, composition_space, geometry, THIS_PATH):
    """
    Returns a VaspEnergyCalculator object, or quits if one cannot be made.

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file

        composition_space: the CompositionSpace of the search

        geometry: the Geometry for the search
    """

    if parameters['EnergyCode']['vasp'] is None:
        if rank_world == 0:
            print('No VASP input files given.')
            comm_world.Abort(rank_world)
    else:
        # the INCAR file
        if 'incar' not in parameters['EnergyCode']['vasp']:
            if rank_world == 0:
                print('An INCAR file must be provided. Please use the "incar" '
                      'keyword.')
                comm_world.Abort(rank_world)
        elif parameters['EnergyCode']['vasp']['incar'] is None:
            if rank_world == 0:
                print('No INCAR file was given after the "incar" keyword. Please '
                      'provide one.')
                comm_world.Abort(rank_world)
        else:
            # get the path to the INCAR file
            incar_path = parameters['EnergyCode']['vasp']['incar']
            # check that the INCAR file exists
            if not os.path.exists(incar_path):
                if rank_world == 0:
                    print('The given INCAR file does not exist.')
                    comm_world.Abort(rank_world)
        # the KPOINTS file
        if 'kpoints' not in parameters['EnergyCode']['vasp']:
            if rank_world == 0:
                print('A KPOINTS file must be provided. Please use the '
                      '"kpoints" keyword.')
                comm_world.Abort(rank_world)
        elif parameters['EnergyCode']['vasp']['kpoints'] is None:
            if rank_world == 0:
                print('No KPOINTS file was given after the "kpoints" keyword. '
                      'Please provide one.')
                comm_world.Abort(rank_world)
        else:
            # get the path to the KPOINTS file
            kpoints_path = parameters['EnergyCode']['vasp']['kpoints']
            # check that the KPOINTS file exists
            if not os.path.exists(kpoints_path):
                if rank_world == 0:
                    print('The given KPOINTS file does not exist.')
                    comm_world.Abort(rank_world)
        # the POTCAR files
        if 'potcars' not in parameters['EnergyCode']['vasp']:
            if rank_world == 0:
                print('POTCAR file(s) must be provided. Please use the '
                      '"potcars" keyword.')
                comm_world.Abort(rank_world)
        elif parameters['EnergyCode']['vasp']['potcars'] is None:
            if rank_world == 0:
                print('No POTCAR files were given after the "potcars" keyword. '
                      'Please provide them.')
                comm_world.Abort(rank_world)
        else:
            # get the the paths to the POTCAR files of each element
            potcar_paths = parameters['EnergyCode']['vasp']['potcars']
            # check that enough POTCAR files have been provided
            elements_list = composition_space.get_all_elements()
            if len(potcar_paths) < len(elements_list):
                if rank_world == 0:
                    print('Not enough POTCAR files provided - one must be '
                          'given for each element in the composition space. '
                          'Please provide them.')
                    comm_world.Abort(rank_world)
            # check that each element has been specified below the
            # 'potcars' keyword
            for element in elements_list:
                if element.symbol not in potcar_paths:
                    if rank_world == 0:
                        print('No POTCAR file given for {}. Please provide '
                              'one.'.format(element.symbol))
                        comm_world.Abort(rank_world)
            # for each element, check that a POTCAR file has been given and
            # that it exists
            for key in potcar_paths:
                if potcar_paths[key] is None:
                    if rank_world == 0:
                        print('No POTCAR file given for {}. Please provide '
                              'one.'.format(key))
                        comm_world.Abort(rank_world)
                elif not os.path.exists(potcar_paths[key]):
                    if rank_world == 0:
                        print('The POTCAR file given for {} does not '
                              'exist.'.format(key))
                        comm_world.Abort(rank_world)

        if 'path_to_callscript' not in parameters['EnergyCode']['vasp']:
            path_to_callscript = THIS_PATH
        elif parameters['EnergyCode']['vasp']['path_to_callscript'] in (None, 'default'):
            path_to_callscript = THIS_PATH
        else:
            path_to_callscript = parameters['EnergyCode']['vasp']['path_to_callscript']

        if 'callscript' not in parameters['EnergyCode']['vasp']:
            callscript = "callvasp"
        elif parameters['EnergyCode']['vasp']['callscript'] in (None, 'default'):
            callscript = "callvasp"
        else:
            callscript = parameters['EnergyCode']['vasp']['callscript']

        return energy_calculators.VaspEnergyCalculator(
            incar_path, kpoints_path, potcar_paths, geometry, path_to_callscript=path_to_callscript,
            callscript=callscript)


def make_stopping_criteria(parameters, composition_space):
    """
    Returns a StoppingCriteria object.

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file

        composition_space: the CompositionSpace of the search
    """

    if 'StoppingCriteria' not in parameters:
        return general.StoppingCriteria(None, composition_space)
    elif parameters['StoppingCriteria'] in (None, 'default'):
        return general.StoppingCriteria(None, composition_space)
    elif 'found_structure' in parameters['StoppingCriteria']:
        if parameters['StoppingCriteria']['found_structure'] in (None,
                                                                 'default'):
            return general.StoppingCriteria(parameters['StoppingCriteria'],
                                            composition_space)
        else:
            # check that the file exists
            given_path = parameters['StoppingCriteria']['found_structure']
            if not os.path.exists(given_path):
                if rank_world == 0:
                    print('The file containing the structure to find does not '
                          'exist.')
                    comm_world.Abort(rank_world)
            # check that the file has the correct suffix or prefix
            elif not (os.path.basename(given_path).endswith('.cif') or
                      os.path.basename(given_path).startswith('POSCAR.')):
                if rank_world == 0:
                    print('File containing structure to find must be in POSCAR or '
                          'cif format and begin with POSCAR. or end with .cif, '
                          'respectively.')
                    comm_world.Abort(rank_world)
            # check that file can be read properly
            else:
                try:
                    Structure.from_file(given_path)
                    return general.StoppingCriteria(
                        parameters['StoppingCriteria'], composition_space)
                except ValueError:
                    if rank_world == 0:
                        print('Error reading the structure to find from the given '
                              'file.')
                        comm_world.Abort(rank_world)
    else:
        return general.StoppingCriteria(parameters['StoppingCriteria'],
                                        composition_space)


def make_variations(parameters, default_fractions, composition_space):
    """
    Creates the variations, using default parameter values if needed.

    Returns a list containing the variation objects (Mating, StructureMut,
    NumAtomssMut and Permutation).

    Args:
        parameters: the dictionary produced by calling yaml.safe_load() on the input
            file

        default_fractions: a dictionary containing the default fractions to use
             for each variation

        composition_space: the CompositionSpace of the search
    """

    if 'Variations' not in parameters:
        return make_default_variations(default_fractions, composition_space)
    elif parameters['Variations'] in (None, 'default'):
        return make_default_variations(default_fractions, composition_space)
    else:
        variations_list = []
        # mating
        if 'Mating' not in parameters['Variations']:
            pass
        elif parameters['Variations']['Mating'] is None:
            if rank_world == 0:
                print('If the "Mating" keyword is used, its "fraction" keyword '
                      'must also be set.')
                comm_world.Abort(rank_world)
        else:
            if parameters['Variations']['Mating']['fraction'] in (None,
                                                                  'default'):
                if rank_world == 0:
                    print('The "fraction" kwyword is not optional and must '
                          'contain a valid entry (between 0 and 1) for the Mating '
                          'variation.')
                    comm_world.Abort(rank_world)
            else:
                mating = variations.Mating(parameters['Variations']['Mating'])
                variations_list.append(mating)

        # structure mutation
        if 'StructureMut' not in parameters['Variations']:
            pass
        elif parameters['Variations']['StructureMut'] is None:
            if rank_world == 0:
                print('If the "StructureMut" keyword is used, its "fraction" '
                      'keyword must also be set.')
                comm_world.Abort(rank_world)
        else:
            if parameters['Variations']['StructureMut']['fraction'] in (
                    None, 'default'):
                if rank_world == 0:
                    print('The "fraction" keyword is not optional and must '
                          'contain a valid entry (between 0 and 1) for the '
                          'StructureMut variation.')
                    comm_world.Abort(rank_world)
            else:
                structure_mut = variations.StructureMut(
                    parameters['Variations']['StructureMut'])
                variations_list.append(structure_mut)

        # mutating the number of atoms in the cell
        if 'NumAtomsMut' not in parameters['Variations']:
            pass
        elif parameters['Variations']['NumAtomsMut'] is None:
            if rank_world == 0:
                print('If the "NumAtomsMut" keyword is used, its "fraction" '
                      'keyword must also be set.')
                comm_world.Abort(rank_world)
        else:
            if parameters['Variations']['NumAtomsMut']['fraction'] in (
                    None, 'default'):
                if rank_world == 0:
                    print('The "fraction" keyword is not optional and must '
                          'contain a valid entry (between 0 and 1) for the '
                          'NumAtomsMut variation.')
                    comm_world.Abort(rank_world)
            else:
                num_atoms_mut = variations.NumAtomsMut(
                    parameters['Variations']['NumAtomsMut'])
                variations_list.append(num_atoms_mut)

        # permutation (swapping atoms)
        if 'Permutation' not in parameters['Variations']:
            pass
        elif parameters['Variations']['Permutation'] is None:
            if rank_world == 0:
                print('If the "Permutation" keyword is used, its "fraction" '
                      'keyword must also be set.')
                comm_world.Abort(rank_world)
        else:
            if parameters['Variations']['Permutation']['fraction'] in (
                    None, 'default'):
                if rank_world == 0:
                    print('The "fraction" keyword is not optional and must '
                          'contain a valid entry (between 0 and 1) for the '
                          'Permutation variation.')
                    comm_world.Abort(rank_world)
            else:
                permutation = variations.Permutation(
                    parameters['Variations']['Permutation'], composition_space)
                variations_list.append(permutation)

        return variations_list


def make_default_variations(default_fractions, composition_space):
    """
    Creates the variations with default parameter values and the provided
    default fractions.

    Returns a list containing the variation objects (Mating, StructureMut,
    NumAtomsMut and Permutation).

    Args:
        default_fractions: a dictionary containing the default fractions to use
             for each variation

        composition_space: the CompositionSpace of the search
    """

    variations_list = []
    mating = variations.Mating({'fraction': default_fractions['mating']})
    structure_mut = variations.StructureMut(
        {'fraction': default_fractions['structure_mut']})
    num_atoms_mut = variations.NumAtomsMut(
        {'fraction': default_fractions['num_atoms_mut']})
    permutation = variations.Permutation(
        {'fraction': default_fractions['permutation']},
        composition_space)
    variations_list.append(mating)
    variations_list.append(structure_mut)
    variations_list.append(num_atoms_mut)
    variations_list.append(permutation)
    return variations_list
