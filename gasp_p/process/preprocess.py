import copy
import os
import sys

import yaml
from gasp.restart.restart import RestartFile
from mpi4py import MPI

from gasp_p import general
from gasp_p import objects_maker
from gasp_p import parameters_printer

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


def load_RESTART(garun_dir):
    thisRestart = None
    if os.path.isdir(garun_dir):
        files = []
        for file in os.listdir(garun_dir):
            if "RESTART_" in file and ".restart" in file:
                thisstr = file.replace("RESTART_", "")
                thisstr = thisstr.replace(".restart", "")
                try:
                    num_calcs = int(thisstr)
                    files.append(num_calcs)
                except:
                    pass
        if len(files) > 0:
            fsorted = sorted(files, reverse=True)
            filename = "RESTART_" + str(fsorted[0]) + ".restart"
            thisRestart = RestartFile.from_file(garun_dir + "/" + filename)
    return thisRestart


def preprocess(THIS_PATH):
    if len(sys.argv) < 2:
        if rank_world == 0:
            print('No input file given.')
            print('Quitting...')
            comm_world.Abort(rank_world)
    else:
        input_file = os.path.abspath(sys.argv[1])
    try:
        with open(input_file, 'r') as f:
            parameters = yaml.safe_load(f)
    except:
        if rank_world == 0:
            print('Error reading input file.')
            print('Quitting...')
            comm_world.Abort(rank_world)

    # make the objects needed by the algorithm
    objects_dict = objects_maker.make_objects(parameters, THIS_PATH)
    # get the objects from the dictionary for convenience
    run_dir_name = objects_dict['run_dir_name']

    # get the path to the run directory - append date and time if
    # the given or default run directory already exists
    garun_dir = str(os.getcwd()) + '/' + run_dir_name
    thisRestart = load_RESTART(garun_dir)
    if thisRestart is not None:
        objects_dict = copy.deepcopy(thisRestart.objects_dict)

    if rank_world == 0:
        # make the run directory and move into it
        os.makedirs(garun_dir, exist_ok=True)
        # make the temp subdirectory where the energy calculations will be done
        os.makedirs(garun_dir + '/temp', exist_ok=True)
        # print the search parameters to a file in the run directory
        parameters_printer.print_parameters(objects_dict, garun_dir)
        # make the data writer
        data_writer = general.DataWriter(garun_dir + '/run_data',
                                         objects_dict['composition_space'], RESTART=thisRestart)
        log_writer = general.LogWriter(THIS_PATH + '/GASP.log',
                                       Screen=objects_dict['Screen'], Log=objects_dict['Log'], RESTART=thisRestart)
    else:
        garun_dir = None
        data_writer = None
        log_writer = None

    garun_dir = comm_world.bcast(garun_dir, root=0)
    data_writer = comm_world.bcast(data_writer, root=0)
    log_writer = comm_world.bcast(log_writer, root=0)

    return objects_dict, garun_dir, data_writer, log_writer, thisRestart
