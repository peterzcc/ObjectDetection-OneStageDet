import os
import subprocess
import time
import signal
import sys


def exec_file(file_name,f_out):
    # current_sp = subprocess.Popen(["zsh", file_name], stderr=f_out, stdout=f_out, )
    if not os.path.isfile(file_name):
        command = ["echo", f"file not exist: {file_name}"]
    else:
        with open(file_name, 'r', encoding="utf8") as f:
            command = f.readline().rstrip().split(" ")
            if command[0] == "":
                command = ["echo", "no jobs"]
    print(f"running: {command}")
    current_sp = subprocess.Popen(command, stderr=f_out, stdout=f_out, )
    return current_sp


def get_git_revision_short_hash(filename, git_root="."):
    return subprocess.check_output(['git', 'log', '-n', '1', '--pretty=format:%H', '--', filename],
                                   cwd=git_root).decode()


def main():
    host_name = os.environ['SLURM_TOPOLOGY_ADDR']
    job_file_name = f"{host_name}.sh"
    git_root = "jobs"
    job_file_path = os.path.join(git_root, job_file_name)
    out_name = f"{job_file_name}.out"
    current_version = get_git_revision_short_hash(job_file_name, git_root)
    print(f"current version: {current_version}")
    f_out = open(out_name, 'w', encoding="utf8")
    current_sp = exec_file(job_file_path, f_out)
    last_check_version_time = time.time()
    while True:
        current_time = time.time()
        if current_time - last_check_version_time > 0.2:
            last_check_version_time = current_time
            latest_version = get_git_revision_short_hash(job_file_name, git_root)
            if latest_version != current_version:
                ret_code = current_sp.poll()
                if ret_code is not None:
                    print(f"current process finished with code {ret_code}, running new version: {latest_version}")
                    f_out = open(out_name, 'w', encoding="utf8")
                    current_sp = exec_file(job_file_path, f_out)
                    current_version = latest_version
                else:
                    if current_sp.poll() is None:
                        current_sp.send_signal(signal.SIGINT)
                        print(f"sent SIGINT to {current_sp.pid}")
                        sig_time = time.time()
                        while current_sp.poll() is None:
                            if time.time() - sig_time > 5.0:
                                print("not finished. sending terminate signal")
                                current_sp.terminate()
                            time.sleep(0.2)
                        current_sp.wait(5.0)
                        f_out.close()

        # time.sleep(0.2)


if __name__ == '__main__':
    main()
