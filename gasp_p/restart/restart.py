import os
import pickle

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


class RestartFile:
    def __init__(
            self,
            progress_index,
            num_finished_calcs,
            garun_dir,
            objects_dict,
            initial_population,
            whole_pop,
    ):
        self.progress_index = progress_index
        self.num_finished_calcs = num_finished_calcs
        self.garun_dir = garun_dir
        self.objects_dict = objects_dict
        self.initial_population = initial_population
        self.whole_pop = whole_pop

    def __str__(self):
        return f"Restart progress_index: {self.progress_index} and finished cals: {self.num_finished_calcs}."

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_file(cls, filename):
        thisRestart = None
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                try:
                    thisRestart = pickle.load(f)
                except:
                    thisRestart = None
        else:
            thisRestart = None
        return thisRestart

    def to_file(self):
        filename = "RESTART_" + str(self.num_finished_calcs) + ".restart"
        filename = self.garun_dir + "/" + filename
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
