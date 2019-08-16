#!/usr/bin/python
'''
    File name: cpu_mem_util.py
    Author: Rajendrakumar Chinnaiyan
    Date created: 12/01/2018
    Python Version: 2.7, 3.6
'''
"""The Module supports VNNI dungeon automation to collect CPU and Memory profiling data using psutil python lib."""

import datetime
import os
import errno
import sys
import time
import argparse
from subprocess import check_output
import ntpath
from emon import EmonCollector

fix_psutil_installation = 2


def install(package):
    import pip as p
    # TODO: Fix this ugly work around to mitigate PIP issue.
    if int(p.__version__.split('.')[0]) > 9:
        from pip._internal import main as pip_main
        pip_main(['install', package])
    else:
        if hasattr(p, 'main'):
            p.main(['install', package])


psutil_not_available = True
# Self installation of required psutil package.
while fix_psutil_installation > 0:
    try:
        import psutil as psu

        fix_psutil_installation = 0
        psutil_not_available = False
    except ImportError:
        print("psutil not found, trying to install and fix it")
        install('psutil')
        fix_psutil_installation -= 1

if psutil_not_available:
    print("\nIts mandatory to have psutil package. Please install using pip 'pip install psutil'\n")
    exit(-1)

# on 0 it will use the average as threshold
SMART_THRESHOLD = 0
MEM_TYPE = ""


class CoreInfo(object):
    core_keys = {"total_cpu": "CPU(s)",
                 "threads_per_core": "Thread(s) per core",
                 "cores_per_socket": "Core(s) per socket",
                 "total_sockets": "Socket(s)"}
    total_cpu = 0
    threads_per_core = 0
    cores_per_socket = 0
    total_sockets = 0

    def __init__(self):
        self.collect_info()

    def collect_info(self):
        n = check_output('lscpu | grep -i -E  "^CPU\(s\):|core|socket"', shell=True).decode('ascii')
        try:
            outputs = n.split("\n")
            for output in outputs:
                if len(output) > 1:
                    output_key = output.split(":")[0].strip()
                    output_value = output.split(":")[1].strip()
                    for key, value in self.core_keys.items():
                        if value in output_key:
                            setattr(self, key, int(output_value))
        except Exception as e:
            print(e)
            raise Exception("Unable to determine HyperThreading status.")

    def summary(self):
        for key, value in self.core_keys.items():
            print("{} : {}".format(key, getattr(self, key)))
        print(self.is_ht_enabled())
        print(self.get_socket_vcpus(0))
        print(self.get_socket_vcpus(1))

    def is_ht_enabled(self):
        if self.threads_per_core == 2:
            return True
        else:
            return False

    def get_socket_vcpus(self, physical_socket):
        vcpus = []
        for vcpu in range(self.total_cpu):
            with open("/sys/devices/system/cpu/cpu{}/topology/physical_package_id".format(vcpu)) as socket_info:
                if int(socket_info.read()) == physical_socket:
                    vcpus.append(vcpu)
        return vcpus


class Measure(object):
    current_cpu_profile = {}
    current_mem_profile = {}
    current_pmem_profile = {}
    coreinfo = None
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    emoncollector = None

    def __init__(self):
        self.coreinfo = CoreInfo()

    def setup(self):

        self.emoncollector = EmonCollector()

        if os.path.isabs(args.file_name):
            log_dir, test_name = ntpath.split(args.file_name)

        if log_dir:
            self.emoncollector.setOutputFilePath(log_dir)

        if test_name:
            if ".log" in test_name:
                test_name = test_name.split(".")[0]
                self.emoncollector.setTimeStamp(test_name)
            else:
                self.emoncollector.setTimeStamp(self.time_stamp)
        else:
            self.emoncollector.setTimeStamp(self.time_stamp)

    def get_log_file_name(self):
        return "profile_log_{}.txt".format(self.time_stamp)

    def profile_with_pid(self, pid, emon, intervel=1.0, delayed=0, duration=0, profile_socket_id=[]):
        if not psu.pid_exists(pid):
            print("Error: Input process ID doesn't seems to running \
             on the system. Please provide a valid process id.\n")
            return
        return self.profile(emon=emon, intervel=intervel, delayed=delayed, duration=duration, pid=pid,
                            profile_socket_id=profile_socket_id)

    def profile_system(self, emon, intervel=1.0, delayed=0, duration=10, profile_socket_id=[]):
        return self.profile(emon, intervel, delayed, duration, profile_socket_id=profile_socket_id)

    def print_cpu_load(self, cpu_load):
        output = ""
        for socket, load in cpu_load.items():
            output += " Socket%s - %.2f%.%" % (socket, load)
        return output

    def get_process_summary(self, pobj):
        res = ""
        pargs = pobj.cmdline()
        if pargs:
            res += "Target process command :\n"
            res += " ".join(pargs)
            res += "\n"
        return res

    def profile(self, emon=False, intervel=1.0, delayed=0, duration=10, pid=None, profile_socket_id=[]):
        global MEM_TYPE
        emon_started = False
        process_summary = None
        if len(profile_socket_id) < 1:
            profile_socket_id = [socket for socket in range(
                self.coreinfo.total_sockets)]
        profile_countdown = 0
        if pid == None:
            pid_check = "True"
        else:
            pobj = psu.Process(int(pid))
            pid_check = "psu.pid_exists({})".format(pid)
            process_summary = self.get_process_summary(pobj)

        if delayed:
            print("waiting for {} seconds".format(delayed))
            time.sleep(delayed)

        if not pid and duration == 0:
            duration = 60
        start = time.time()
        _timeout = time.time() + duration
        if not ("emon" in (p.name() for p in psu.process_iter())) and emon:
            self.emoncollector.start()
            emon_started = True
        while eval(pid_check):
            sys.stdout.write("\rProfiling %d seconds" % (profile_countdown + 1))
            sys.stdout.flush()
            profile_countdown_s = datetime.datetime.now().strftime("%Y %m %d %H:%M:%S")
            if args.verbose:
                print("")
            load = self.profile_socket(profile_socket_id)
            if args.verbose:
                sys.stdout.write("   CPU Load = ")
                print(self.print_cpu_load(load))
            if pid:
                pmem = self.profile_process_memory(pobj)
                self.current_pmem_profile[profile_countdown_s] = pmem
                if args.verbose:
                    print("   Process {} rss memory : {} MB".format(pid, pmem))

            smem = self.profile_system_memory()
            if args.verbose:
                print("   Overall System used memory : {} MB".format(smem))

            self.current_cpu_profile[profile_countdown_s] = load
            self.current_mem_profile[profile_countdown_s] = smem

            time.sleep(intervel)
            if duration and (time.time() > _timeout):
                break
            profile_countdown += 1
        if emon_started:
            self.emoncollector.stop()
        print("\n")
        duration = int(time.time() - start)
        avg_cpu_load, smart_load = self.average_cpuload(
            self.current_cpu_profile)
        summary_str = ("Profiling completed for {} seconds\n{}".format(duration, ('*' * 65)))
        summary_str += "\nTotal Sockets     \t\t: {}".format(self.coreinfo.total_sockets)
        summary_str += "\nHT Enabled        \t\t: {}".format(self.coreinfo.is_ht_enabled())
        summary_str += "\nPhysical Cores Per socket \t: {}".format(self.coreinfo.cores_per_socket)
        summary_str += "\nLogical Cores Per socket \t: {}".format(
            self.coreinfo.cores_per_socket * self.coreinfo.threads_per_core)
        summary_str += "\nThreads Per core  \t\t: {}".format(self.coreinfo.threads_per_core)
        summary_str += "\nAvg CPU Utilization         \t:"
        summary_str += self.print_cpu_load(avg_cpu_load) + "\n"
        summary_str += "Avg CPU Utilization (> avg)   \t:"
        summary_str += self.print_cpu_load(smart_load) + "\n"
        avg_mem_usage = self.average_memusage(self.current_mem_profile)
        summary_str += "Overall System memory usage \t: {} MB \n".format(avg_mem_usage)
        if pid:
            avg_pmem_usage = self.average_memusage(self.current_pmem_profile)
            summary_str += "{} Process RSS memory usage \t: {} MB \n".format(pid, avg_pmem_usage)
        summary_str += "{} \n".format(('*' * 65))
        if process_summary:
            summary_str += "{} \n".format(process_summary)

        print(summary_str)
        if args.file_name:
            self.write_to_file(args.file_name, summary_str)

    def write_to_file(self, file_name, summary):
        with open(file_name, 'w') as log_file:
            countdown = self.current_cpu_profile.keys()
            for count in countdown:
                line_str = "{} ".format(count)
                for socket, load in self.current_cpu_profile.get(count).items():
                    line_str += "\tSocket%s - %.2f%.% " % (socket, load)
                if len(self.current_pmem_profile) > 0:
                    line_str += "\t Process RSS memory usage  : {} MB".format(self.current_pmem_profile.get(count))
                line_str += "\t Overall System memory usage : {} MB\n".format(self.current_mem_profile.get(count))
                log_file.write(line_str)
            log_file.write(summary)
            print("Logs located at : {}".format(file_name))

    def average_cpuload(self, total_data):
        global SMART_THRESHOLD
        console_map = {}
        smart_avg = {}
        for util_map in total_data.values():
            sockets = util_map.keys()
            for socket in sockets:
                _value = util_map.get(socket)
                if socket in console_map:
                    console_map[socket].append(_value)
                else:
                    console_map[socket] = [_value]

        for key in console_map.keys():
            overall_avg = sum(console_map[key]) / len(console_map[key])
            if SMART_THRESHOLD > 0 and SMART_THRESHOLD <= 100:
                smart_numbers = [x for x in console_map[key]
                                 if x > SMART_THRESHOLD]
            else:
                smart_numbers = [
                    x for x in console_map[key] if x > overall_avg]
            console_map[key] = round(overall_avg, 2)
            if len(smart_numbers) > 0:
                smart_avg[key] = round(
                    sum(smart_numbers) / len(smart_numbers), 2)
            else:
                smart_avg[key] = round(overall_avg, 2)
        return console_map, smart_avg

    def average_memusage(self, total_data):
        overall_avg = round(sum(total_data.values()) /
                            len(total_data.values()), 2)
        return overall_avg

    def profile_process_memory(self, pobj):
        prss = 0
        if pobj:
            with pobj.oneshot():
                prss = round((int(pobj.memory_info().rss) / (1024 * 1024)), 2)
        return prss

    def profile_system_memory(self, clean_cache=False):
        if clean_cache:
            output = check_output('sync; echo 3 > /proc/sys/vm/drop_caches', shell=True).decode('ascii')
        return round((int(psu.virtual_memory().used) / (1024 * 1024)), 2)

    def profile_socket(self, socket_id):
        load = None
        cpu_load = psu.cpu_percent(percpu=True)
        if type(socket_id) is list:
            if len(socket_id) > self.coreinfo.total_sockets:
                raise Exception("The system seems to have lesser \
                 number of sockets than requested sockets")
            load = {}
            for socket in socket_id:
                if socket > (self.coreinfo.total_sockets):
                    raise Exception(
                        "The system seems to have lesser number \
                         of sockets than requested sockets")
                vcpus = self.coreinfo.get_socket_vcpus(socket)
                agg_cpu_load = round(sum([cpu_load[_l]
                                          for _l in vcpus]) / len(vcpus), 2)
                load[socket] = agg_cpu_load
        else:
            if socket_id > (self.coreinfo.total_sockets - 1):
                raise Exception(
                    "The system seems to have lesser number of \
                    sockets than requested sockets")
            vcpus = self.coreinfo.get_socket_vcpus(socket_id)
            agg_cpu_load = round(sum([cpu_load[_l]
                                      for _l in vcpus]) / len(vcpus), 2)
            load = agg_cpu_load
        return load


parser = argparse.ArgumentParser(description='Process commandline arguments')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-p', '--pid', default=None, type=int)
parser.add_argument('-d', '--duration', default=0, type=int)
parser.add_argument('-i', '--intervel', default=1, type=int)
parser.add_argument('-l', '--delayed', default=0, type=int)
parser.add_argument('-s', '--socket_id', default=[], type=list)
parser.add_argument('-f', '--file_name', default=None,
                    const="profilelog.txt", type=str, nargs='?')
parser.add_argument('-t', '--smart_threshold', default=0, type=int)
# parser.add_argument('-e', '--emon_run', default=False, type=boolean)
parser.add_argument('-e', dest='emon', action='store_true', default=False, help='Run Emon with psutil')
parser.add_argument('-z', '--zdir', default='/var/log/vnni_dungeon/profile_logs', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    measure = Measure()
    if args.verbose: print(args)
    SMART_THRESHOLD = args.smart_threshold
    MEM_TYPE = "Overall System memory"

    if not args.file_name:  # and (args.file_name == "profilelog.txt"):
        args.file_name = measure.get_log_file_name()

    if args.zdir:
        if not os.path.isdir(args.zdir):
            os.mkdir(args.zdir)

    if args.file_name:
        args.file_name = os.path.join(args.zdir, args.file_name)

    if len(args.socket_id) > 0:
        args.socket_id = map(int, args.socket_id)

    if args.smart_threshold not in range(0, 101):
        print("\n Error: Smart threshold must be withtin 1-100. \
        If you are not sure why to use this option, just disable it\n")
        exit(0)
    measure.setup()
    if args.pid:
        if args.duration == 0:
            print("\n*** Warning: Samples will be collected for target process entier duration. \
                   \n*** This is beacuse you have not setup duration with process ID \
                    \n*** Tip: Take a look at duration option to limit the profiling time. \n")
        measure.profile_with_pid(args.pid, args.emon, intervel=args.intervel, delayed=args.delayed,
                                 duration=args.duration, profile_socket_id=args.socket_id)
    else:
        if args.duration == 0:
            args.duration = 10
        print("Info: Profiling started... Samples will be collected for \
         {} seconds duration".format(args.duration))
        measure.profile_system(args.emon, intervel=args.intervel, delayed=args.delayed,
                               duration=args.duration, profile_socket_id=args.socket_id)




























































































































































































































































































































































































































































































