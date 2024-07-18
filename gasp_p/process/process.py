import copy
import os
import random

import numpy as np
from mpi4py import MPI

import gasp_p.mpiconf.MPIconf as mympi
from gasp_p import general
from gasp_p import population
from gasp_p.mpiconf.MPIconf import MPI_Tags
from gasp_p.restart.restart import RestartFile

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


def master_creat_organisms(objects_dict, garun_dir, data_writer, log_writer, THIS_PATH,
                           whole_pop, num_finished_calcs, initial_population,
                           creator, stopping_criteria, ntask_tot):
    status = MPI.Status()
    composition_space = objects_dict['composition_space']
    constraints = objects_dict['constraints']
    geometry = objects_dict['geometry']
    developer = objects_dict['developer']
    redundancy_guard = objects_dict['redundancy_guard']
    pool = objects_dict['pool']
    id_generator = objects_dict['id_generator']

    processing_tasks = np.arange(ntask_tot, dtype=int)
    nproc_task = objects_dict['nproc_task']

    task_index = 0
    completed_tasks = 0
    working_jobs = []
    while completed_tasks < ntask_tot:
        idtask = comm_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        s = status.Get_source()
        tag = status.Get_tag()
        thiscolor = int((s - 1) / nproc_task)
        ##rank_local = (s-1)%nproc_task
        if thiscolor in working_jobs:
            if tag == MPI_Tags.DONE:
                relaxed_organisms = comm_world.recv(source=s, tag=51)
                logstr_dicts = comm_world.recv(source=s, tag=52)
                relaxed_organism = relaxed_organisms[thiscolor]
                logstr = logstr_dicts[thiscolor]
                log_writer.write_data(logstr)
                num_finished_calcs += 1

                if num_finished_calcs % objects_dict['num_calcs_to_restartfile'] == 0:
                    thisRestart = RestartFile(0, num_finished_calcs,
                                              garun_dir, objects_dict,
                                              initial_population, whole_pop)
                    thisRestart.to_file()
                # take care of relaxed organism
                if relaxed_organism is not None:
                    geometry.unpad(relaxed_organism.cell, constraints)
                    if developer.develop(relaxed_organism,
                                         composition_space,
                                         constraints, geometry, pool, log_writer):
                        redundant_organism = \
                            redundancy_guard.check_redundancy(
                                relaxed_organism, whole_pop, geometry, log_writer)
                        if redundant_organism is not None:  # redundant
                            if redundant_organism.is_active and \
                                    redundant_organism.epa > \
                                    relaxed_organism.epa:
                                initial_population.replace_organism(
                                    redundant_organism,
                                    relaxed_organism,
                                    composition_space, log_writer)
                                progress = \
                                    initial_population.get_progress(
                                        composition_space)
                                data_writer.write_data(
                                    relaxed_organism,
                                    num_finished_calcs, progress)
                                logstr = f'Number of energy calculations so far: {num_finished_calcs}'
                                log_writer.write_data(logstr)
                        else:  # not redundant
                            stopping_criteria.check_organism(
                                relaxed_organism, redundancy_guard,
                                geometry)
                            initial_population.add_organism(
                                relaxed_organism, composition_space, log_writer)
                            whole_pop.append(relaxed_organism)
                            progress = \
                                initial_population.get_progress(
                                    composition_space)
                            data_writer.write_data(
                                relaxed_organism, num_finished_calcs,
                                progress)
                            logstr = f'Number of energy calculations so far: {num_finished_calcs}'
                            log_writer.write_data(logstr)
                            if creator.is_successes_based and \
                                    relaxed_organism.made_by == creator.name:
                                creator.update_status()
                working_jobs.remove(thiscolor)
                if task_index < ntask_tot: completed_tasks += 1
        else:
            if tag == MPI_Tags.READY:
                if task_index < ntask_tot:
                    idtask = processing_tasks[task_index]

                    isValid = True
                    while isValid:
                        new_organism = creator.create_organism(
                            id_generator, composition_space, constraints, log_writer, random)
                        while new_organism is None:
                            new_organism = creator.create_organism(
                                id_generator, composition_space, constraints, log_writer, random)
                        if new_organism is not None:  # loop above could return None
                            geometry.unpad(new_organism.cell, constraints)
                            if developer.develop(new_organism, composition_space,
                                                 constraints, geometry, pool, log_writer):
                                redundant_organism = redundancy_guard.check_redundancy(
                                    new_organism, whole_pop, geometry, log_writer)
                                if redundant_organism is None:  # no redundancy
                                    # add a copy to whole_pop so the organisms in
                                    # whole_pop don't change upon relaxation
                                    whole_pop.append(copy.deepcopy(new_organism))
                                    geometry.pad(new_organism.cell)
                                    stopping_criteria.update_calc_counter()

                                    comm_world.send(idtask, dest=s, tag=MPI_Tags.START)
                                    comm_world.send(new_organism, dest=s, tag=11)

                                    working_jobs.append(thiscolor)
                                    task_index += 1

                                    isValid = False
                else:
                    comm_world.send(None, dest=s, tag=MPI_Tags.EXIT)
            elif tag == MPI_Tags.EXIT:
                completed_tasks += 1

    return whole_pop, num_finished_calcs, initial_population, creator, stopping_criteria


def slave_creat_organisms(objects_dict, garun_dir, log_writer, thiscolor, comm_split):
    status = MPI.Status()
    rank_local = comm_split.Get_rank()
    nproc_task = objects_dict['nproc_task']
    energy_calculator = objects_dict['energy_calculator']
    composition_space = objects_dict['composition_space']
    while True:
        if rank_local == 0:
            comm_world.send(None, dest=0, tag=MPI_Tags.READY)
            idtask = comm_world.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == MPI_Tags.START:
                new_organism = comm_world.recv(source=0, tag=11)
            else:
                new_organism = None
        else:
            idtask = None
            new_organism = None
            tag = None

        if nproc_task > 1:
            tag = comm_split.bcast(tag, root=0)
            idtask = comm_split.bcast(idtask, root=0)
            new_organism = comm_split.bcast(new_organism, root=0)

        if tag == MPI_Tags.START:
            relaxed_organisms = {}
            logstr_dicts = {}
            energy_calculator.do_energy_calculation(new_organism, relaxed_organisms, logstr_dicts, thiscolor,
                                                    composition_space, nproc=nproc_task, comm=comm_split)
            if rank_local == 0:
                comm_world.send(idtask, dest=0, tag=MPI_Tags.DONE)
                comm_world.send(relaxed_organisms, dest=0, tag=51)
                comm_world.send(logstr_dicts, dest=0, tag=52)
            else:
                pass
        elif tag == MPI_Tags.EXIT:
            break
    if rank_local == 0: comm_world.send(None, dest=0, tag=MPI_Tags.EXIT)


def creat_organisms(objects_dict, garun_dir, data_writer, log_writer, THIS_PATH,
                    whole_pop, num_finished_calcs, initial_population,
                    ntask_time, thiscolor, comm_split):
    stopping_criteria = objects_dict['stopping_criteria']
    for creator in objects_dict['organism_creators']:
        if rank_world == 0:
            logstr = f'Making {creator.number} organisms with {creator.name}'
            log_writer.write_data(logstr)

        while not creator.is_finished and not stopping_criteria.are_satisfied:
            ntask_tot = creator.number - creator.num_made
            if ntask_tot > 0:
                this_ntask_time = min(ntask_tot, ntask_time)
                if rank_world == 0:
                    whole_pop, num_finished_calcs, initial_population, creator, stopping_criteria = master_creat_organisms(
                        objects_dict, garun_dir, data_writer, log_writer, THIS_PATH,
                        whole_pop, num_finished_calcs, initial_population,
                        creator, stopping_criteria, this_ntask_time)
                else:
                    if thiscolor < this_ntask_time:
                        slave_creat_organisms(objects_dict, garun_dir, log_writer, thiscolor, comm_split)
                    else:
                        pass

                comm_world.Barrier()
                if rank_world == 0:
                    pass
                else:
                    stopping_criteria = None
                    creator = None

                stopping_criteria = comm_world.bcast(stopping_criteria, root=0)
                creator = comm_world.bcast(creator, root=0)

        comm_world.Barrier()
        if rank_world == 0:
            pass
        else:
            #stopping_criteria = None
            stopping_criteria = None
            creator = None

        stopping_criteria = comm_world.bcast(stopping_criteria, root=0)
        creator = comm_world.bcast(creator, root=0)

    comm_world.Barrier()
    if rank_world == 0:
        pass
    else:
        #stopping_criteria = None
        composition_space = None
        whole_pop = []
        num_finished_calcs = None
        #initial_population = None

    #stopping_criteria = comm_world.bcast(stopping_criteria, root=0)
    composition_space = comm_world.bcast(objects_dict['composition_space'], root=0)
    #whole_pop = comm_world.bcast(whole_pop, root=0)
    num_finished_calcs = comm_world.bcast(num_finished_calcs, root=0)
    #initial_population = comm_world.bcast(initial_population, root=0)
    if rank_world == 0:
        pass
    else:
        objects_dict['composition_space'] = composition_space
    return objects_dict, whole_pop, num_finished_calcs, initial_population


def master_creat_offsprings(objects_dict, garun_dir, data_writer, log_writer, THIS_PATH,
                            whole_pop, num_finished_calcs, initial_population,
                            offspring_generator, stopping_criteria,
                            ntask_tot):
    status = MPI.Status()

    composition_space = objects_dict['composition_space']
    constraints = objects_dict['constraints']
    geometry = objects_dict['geometry']
    developer = objects_dict['developer']
    redundancy_guard = objects_dict['redundancy_guard']
    pool = objects_dict['pool']
    id_generator = objects_dict['id_generator']
    variations = objects_dict['variations']

    processing_tasks = np.arange(ntask_tot, dtype=int)
    nproc_task = objects_dict['nproc_task']

    task_index = 0
    completed_tasks = 0
    working_jobs = []
    while completed_tasks < ntask_tot:
        idtask = comm_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        s = status.Get_source()
        tag = status.Get_tag()
        thiscolor = int((s - 1) / nproc_task)
        ##rank_local = (s-1)%nproc_task
        if thiscolor in working_jobs:
            if tag == MPI_Tags.DONE:
                relaxed_organisms = comm_world.recv(source=s, tag=51)
                logstr_dicts = comm_world.recv(source=s, tag=52)
                relaxed_offspring = relaxed_organisms[thiscolor]
                logstr = logstr_dicts[thiscolor]
                log_writer.write_data(logstr)
                num_finished_calcs += 1

                if num_finished_calcs % objects_dict['num_calcs_to_restartfile'] == 0:
                    thisRestart = RestartFile(1, num_finished_calcs,
                                              garun_dir, objects_dict,
                                              initial_population, whole_pop)
                    thisRestart.to_file()

                # take care of relaxed organism
                if relaxed_offspring is not None:
                    geometry.unpad(relaxed_offspring.cell, constraints)
                    if developer.develop(relaxed_offspring,
                                         composition_space,
                                         constraints, geometry, pool, log_writer):
                        redundant_organism = \
                            redundancy_guard.check_redundancy(
                                relaxed_offspring, whole_pop, geometry, log_writer)
                        if redundant_organism is not None:  # redundant
                            if redundant_organism.epa > relaxed_offspring.epa:
                                pool.replace_organism(redundant_organism,
                                                      relaxed_offspring,
                                                      composition_space, log_writer)
                                pool.compute_fitnesses()
                                pool.compute_selection_probs()
                                pool.print_summary(composition_space, num_finished_calcs, log_writer)
                                progress = pool.get_progress(composition_space)
                                data_writer.write_data(relaxed_offspring,
                                                       num_finished_calcs,
                                                       progress)
                                logstr = f'Number of energy calculations so far: {num_finished_calcs}'
                                log_writer.write_data(logstr)
                        else:  # not redundant
                            redundant_organism = \
                                redundancy_guard.check_redundancy(
                                    relaxed_offspring, whole_pop, geometry, log_writer)

                        if redundant_organism is None:  # not redundant
                            stopping_criteria.check_organism(
                                relaxed_offspring, redundancy_guard, geometry)
                            pool.add_organism(relaxed_offspring,
                                              composition_space, log_writer)
                            whole_pop.append(relaxed_offspring)

                            # check if we've added enough new offspring
                            # organisms to the pool that we can remove the
                            # initial population organisms from the front
                            # (right end) of the queue.
                            if pool.num_adds == pool.size:
                                logstr = f'Removing the initial population from the pool'
                                log_writer.write_data(logstr)
                                for _ in range(len(
                                        initial_population.initial_population)):
                                    removed_org = pool.queue.pop()
                                    removed_org.is_active = False
                                    logstr = f'Removing organism {removed_org.id} from the pool'
                                    log_writer.write_data(logstr)

                            # if the initial population organisms have already
                            # been removed from the pool's queue, then just
                            # need to pop one organism from the front (right
                            # end) of the queue.
                            elif pool.num_adds > pool.size:
                                removed_org = pool.queue.pop()
                                removed_org.is_active = False
                                logstr = f'Removing organism {removed_org.id} from the pool'
                                log_writer.write_data(logstr)

                            pool.compute_fitnesses()
                            pool.compute_selection_probs()
                            pool.print_summary(composition_space, num_finished_calcs, log_writer)
                            progress = pool.get_progress(composition_space)
                            data_writer.write_data(relaxed_offspring,
                                                   num_finished_calcs,
                                                   progress)
                            logstr = f'Number of energy calculations so far: {num_finished_calcs}'
                            log_writer.write_data(logstr)
                working_jobs.remove(thiscolor)
                if task_index < ntask_tot: completed_tasks += 1
        else:
            if tag == MPI_Tags.READY:
                if task_index < ntask_tot:
                    idtask = processing_tasks[task_index]
                    unrelaxed_offspring = offspring_generator.make_offspring_organism(
                        random, pool, variations, geometry, id_generator, whole_pop,
                        developer, redundancy_guard, composition_space, constraints, log_writer)
                    whole_pop.append(copy.deepcopy(unrelaxed_offspring))
                    geometry.pad(unrelaxed_offspring.cell)
                    stopping_criteria.update_calc_counter()

                    comm_world.send(idtask, dest=s, tag=MPI_Tags.START)
                    comm_world.send(unrelaxed_offspring, dest=s, tag=11)

                    working_jobs.append(thiscolor)
                    task_index += 1
                else:
                    comm_world.send(None, dest=s, tag=MPI_Tags.EXIT)
            elif tag == MPI_Tags.EXIT:
                completed_tasks += 1

    return whole_pop, num_finished_calcs, stopping_criteria


def slave_creat_offsprings(objects_dict, garun_dir, log_writer, thiscolor, comm_split):
    status = MPI.Status()
    rank_local = comm_split.Get_rank()
    nproc_task = objects_dict['nproc_task']
    energy_calculator = objects_dict['energy_calculator']
    composition_space = objects_dict['composition_space']
    while True:
        if rank_local == 0:
            comm_world.send(None, dest=0, tag=MPI_Tags.READY)
            idtask = comm_world.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == MPI_Tags.START:
                unrelaxed_offspring = comm_world.recv(source=0, tag=11)
            else:
                unrelaxed_offspring = None
        else:
            idtask = None
            unrelaxed_offspring = None
            tag = None

        if nproc_task > 1:
            tag = comm_split.bcast(tag, root=0)
            idtask = comm_split.bcast(idtask, root=0)
            unrelaxed_offspring = comm_split.bcast(unrelaxed_offspring, root=0)

        if tag == MPI_Tags.START:
            relaxed_organisms = {}  #placeholder
            logstr_dicts = {}
            energy_calculator.do_energy_calculation(unrelaxed_offspring, relaxed_organisms, logstr_dicts, thiscolor,
                                                    composition_space, nproc=nproc_task, comm=comm_split)
            if rank_local == 0:
                comm_world.send(idtask, dest=0, tag=MPI_Tags.DONE)
                comm_world.send(relaxed_organisms, dest=0, tag=51)
                comm_world.send(logstr_dicts, dest=0, tag=52)
            else:
                pass
        elif tag == MPI_Tags.EXIT:
            break
    if rank_local == 0: comm_world.send(None, dest=0, tag=MPI_Tags.EXIT)


def create_offsprings(objects_dict, garun_dir, data_writer, log_writer, THIS_PATH,
                      whole_pop, num_finished_calcs, initial_population,
                      offspring_generator,
                      ntask_time, thiscolor, comm_split):
    stopping_criteria = objects_dict['stopping_criteria']
    while not stopping_criteria.are_satisfied:
        ntask_tot = ntask_time
        this_ntask_time = min(ntask_tot, ntask_time)
        if rank_world == 0:
            whole_pop, num_finished_calcs, stopping_criteria = master_creat_offsprings(objects_dict, garun_dir,
                                                                                       data_writer, log_writer,
                                                                                       THIS_PATH,
                                                                                       whole_pop, num_finished_calcs,
                                                                                       initial_population,
                                                                                       offspring_generator,
                                                                                       stopping_criteria,
                                                                                       this_ntask_time)
        else:
            if thiscolor < this_ntask_time:
                slave_creat_offsprings(objects_dict, garun_dir, log_writer, thiscolor, comm_split)
            else:
                pass

        comm_world.Barrier()
        if rank_world == 0:
            pass
        else:
            stopping_criteria = None
        stopping_criteria = comm_world.bcast(stopping_criteria, root=0)

    comm_world.Barrier()
    if rank_world == 0:
        pass
    else:
        stopping_criteria = None
        objects_dict['composition_space'] = None
        whole_pop = []
        num_finished_calcs = None

    stopping_criteria = comm_world.bcast(stopping_criteria, root=0)
    composition_space = comm_world.bcast(objects_dict['composition_space'], root=0)
    num_finished_calcs = comm_world.bcast(num_finished_calcs, root=0)

    if rank_world == 0:
        pass
    else:
        objects_dict['composition_space'] = composition_space

    return objects_dict, whole_pop, num_finished_calcs


def process(objects_dict, garun_dir, data_writer, log_writer, thisRestart, THIS_PATH):
    nproc_task = objects_dict["nproc_task"]
    start_proc = 1
    if nproc_task == size_world:
        if rank_world == 0:
            print(f"Number of process must be greater than {nproc_task}!")
            comm_world.Abort(rank_world)

    ntask_time = mympi.get_ntask_time(nproc_task, start_proc=start_proc, thiscomm=None)
    comm_split, thiscolor = mympi.split_communicator(nproc_task, start_proc=start_proc, thiscomm=None)

    if thisRestart is None:
        whole_pop = []
        num_finished_calcs = 0
        initial_population = population.InitialPopulation(objects_dict['run_dir_name'])
        progress_index = 0
    else:
        whole_pop = copy.deepcopy(thisRestart.whole_pop)
        num_finished_calcs = thisRestart.num_finished_calcs
        initial_population = copy.deepcopy(thisRestart.initial_population)
        progress_index = thisRestart.progress_index
        thisRestart = None

    os.chdir(garun_dir)
    if progress_index == 0:
        objects_dict, whole_pop, num_finished_calcs, initial_population = creat_organisms(objects_dict, garun_dir,
                                                                                          data_writer, log_writer,
                                                                                          THIS_PATH,
                                                                                          whole_pop, num_finished_calcs,
                                                                                          initial_population,
                                                                                          ntask_time, thiscolor,
                                                                                          comm_split)

        #os.chdir(THIS_PATH)
        if rank_world == 0:
            objects_dict['pool'].add_initial_population(initial_population, objects_dict['composition_space'],
                                                        log_writer)
        comm_world.Barrier()

    offspring_generator = general.OffspringGenerator()
    os.chdir(garun_dir)
    objects_dict, whole_pop, num_finished_calcs = create_offsprings(objects_dict, garun_dir, data_writer, log_writer,
                                                                    THIS_PATH,
                                                                    whole_pop, num_finished_calcs, initial_population,
                                                                    offspring_generator,
                                                                    ntask_time, thiscolor, comm_split)

    comm_world.Barrier()
    comm_split.Free()
