import os
import logging
import random
import time
import multiprocessing
import signal

from functools import partial

from tpunicorn.tpu import get_tpu
from tpunicorn.program import is_preempted, recreate

NAMES = ["wheatley", "chonk", "gerard", "simon", "goose", "megatron", "james", "jill",
         "anna", "pietro"]


class Timeout:

    def __init__(self, seconds=1, error_message='TimeoutError'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class TPUMaker:

    def __init__(self, project=None, zone=None, tf_version=None, names=None, debug_mode=True, preemptible_v8s=False):
        # set defaults
        self.namelist = NAMES if names is None else names
        self.tf_version = "1.15.2" if tf_version is None else tf_version
        self.preemptible_v8s = preemptible_v8s
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG) if debug_mode else self.logger.setLevel(logging.INFO)
        self.project = project
        self.zone = zone

    def make_tpu(self, size, name=None, tf_version=None, accelerator_type="v3", preemptible=True, zone=None,
                 project=None):

        project = self.project if project is None else project
        assert project is not None, "Please set default project with maketpu setproject projectname, or pass in a " \
                                    "project to maketpu.\n e.g, maketpu test 8 --project <projectname>"

        zone = self.zone if zone is None else zone
        assert zone is not None, "Please set default zone with maketpu setzone zonename, or pass in a " \
                                 "zone to maketpu.\n e.g, maketpu test 8 --zone <zonename>"

        # if making a v-8, set preemptible to false if this is the project's default
        if not self.preemptible_v8s and size == 8:
            preemptible = False
            self.logger.debug(
                "Setting preemptible to false, as this project's does not have access to preemptible v-8s")
        if preemptible:
            p = "--preemptible"
        else:
            p = ""

        # if no name is specified, pick one at random from the namelist
        if name is None:
            name = self.get_name()

        tf_version = self.tf_version if tf_version is None else tf_version

        command = f"gcloud compute tpus create {name} --zone {zone} --project {project} --network default --version {tf_version} --accelerator-type {accelerator_type}-{size} {p}"
        self.logger.info(command)
        os.system(command)

    def recreate_tpu(self, name, zone=None, project=None, tf_version=None, preempted=True, retry=60,
                     retry_randomness=1.5):
        recreate(name, self.zone if zone is None else zone,
                 self.project if project is None else project,
                 self.tf_version if tf_version is None else tf_version,
                 yes=True, dry_run=False, preempted=preempted,
                 command=[], retry=retry, retry_randomness=retry_randomness)

    def add_to_namelist(self, name):
        self.namelist.append(name)

    def set_project(self, project_name):
        self.project = project_name

    def set_zone(self, zone):
        self.zone = zone

    def tpu_exists(self, name):
        if get_tpu(name, project=self.project, silent=True) is None:
            return False
        else:
            return True

    def is_preempted(self, tpu):
        return is_preempted(tpu, zone=self.zone, project=self.project)

    def get_name(self):
        self.logger.debug("getting name")
        if self.namelist:
            available_names = self.namelist
            name = random.choice(available_names)
            x = 0
            while True:
                x += 1
                self.logger.debug(available_names)
                if not available_names:
                    raise Exception("All tpu names in default namelist already exist - please pass a name to "
                                    "maketpu or update default namelist")
                if self.tpu_exists(name):
                    self.logger.debug(f'TPU {name} exists')
                    available_names.remove(name)
                    name = random.choice(available_names)
                    self.logger.debug(f'trying {name}')
                else:
                    break
            self.logger.debug(f"got name {name}")
            return name
        else:
            raise Exception("No name specified and default namelist is empty")


def test_fn(x):
    print('hello world')
    time.sleep(10)


class TPUKeepAlive(TPUMaker):
    """
    Runs a python script or function that runs on TPUs, creating a tpu before running, and remaking and rerunning
    the function if the tpu gets preempted.
    """

    def __init__(self, size, name, project, zone, tf_version, *args, wait_time=60, restart_after=86400, **kwargs):
        super().__init__(project, zone, tf_version, *args, **kwargs)
        self.size, self.name, self.wait_time = size, name, wait_time
        self.restart_after = restart_after

    def run_script(self, cmd):
        fn = partial(os.system, command=cmd)
        self.run_fn(fn, except_error=Exception)

    def _run_fn(self, fn, *args, on_timeout_fn, except_error=Exception, **kwargs):
        timeout_message = f'Restarting TPU - {self.restart_after / (60 ** 2)} hours have passed.'
        finished = True
        while True:
            with Timeout(seconds=self.restart_after, error_message=timeout_message):
                try:
                    fn(*args, **kwargs)
                except TimeoutError as e:
                    print(e)
                    on_timeout_fn()
                    finished = False
                except except_error as e:
                    print(f'\nError raised from process: ')
                    print(e)
                    # self.recreate_tpu(name=self.name, preempted=True, retry=self.wait_time)
                    finished = True
            if finished:
                break

    def run_fn(self, fn, *args, except_error=Exception, **kwargs):
        if not self.tpu_exists(self.name):
            self.make_tpu(self.size, self.name)
        recreate_fn = partial(self.recreate_tpu, name=self.name, preempted=True, retry=self.wait_time)
        kwargs.update({"except_error": except_error,
                       "on_timeout_fn": recreate_fn})
        while True:
            main_fn = multiprocessing.Process(target=self._run_fn, args=(fn, *args), kwargs=kwargs)
            main_fn.start()
            while True:
                # periodically check if TPU is preempted
                if not main_fn.is_alive():
                    # if the process is finished, we want to exit the loop
                    finished = True
                    break
                time.sleep(self.wait_time)
                if self.is_preempted(self.name):
                    main_fn.terminate()
                    recreate_fn()
                    finished = False
                    break
            if finished:
                break


class TPUKeepAliveTest(TPUKeepAlive):

    def run_fn_test(self, except_error=Exception, **kwargs):
        if not False:
            print(f'making tpu {self.name}...')
        timeout_fn = partial(print, f'Recreating tpu {self.name}...')
        kwargs.update({"except_error": except_error,
                       "on_timeout_fn": timeout_fn})
        while True:
            main_fn = multiprocessing.Process(target=self._run_fn, args=(test_fn, None), kwargs=kwargs)
            main_fn.start()
            while True:
                # periodically check if TPU is preempted
                if not main_fn.is_alive():
                    # if the process is finished, we want to exit the loop
                    finished = True
                    break
                time.sleep(self.wait_time)
                if random.random() < 0.05:
                    print('Fake tpu is pre-empted! Terminating fn')
                    main_fn.terminate()
                    time.sleep(1)
                    timeout_fn()
                    finished = False
                    break
            if finished:
                break


if __name__ == "__main__":
    # test recreate on preemption
    t = TPUKeepAliveTest(8, 'test', None, None, None, wait_time=1, restart_after=11)
    t.run_fn_test()

    # test recreate on timeout
    t = TPUKeepAliveTest(8, 'test', None, None, None, wait_time=1, restart_after=9)
    t.run_fn_test()

