from mpi4py import MPI

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


def postprocess(objects_dict, garun_dir, data_writer, log_writer, THIS_PATH):
    pass
#shutil.rmtree(os.path.join(garun_dir, "temp"))
