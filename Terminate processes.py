import os
import psutil
import shutil

folder_path = 'D:/image'
lock_files_to_delete = []  # Store lock file paths to be deleted

# Terminate processes using the lock file
def terminate_process_using_lock_file(lock_file):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for file in proc.open_files():
                if lock_file in file.path:
                    print(f"Terminating process {proc.pid} - {proc.info}")
                    proc.terminate()  # Terminate the process
        except Exception as e:
            pass

# Traverse folder structure, find and terminate processes using lock files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        lock_file = os.path.join(root, file)
        if lock_file.endswith('.lock'):
            lock_files_to_delete.append(lock_file)

for lock_file in lock_files_to_delete:
    try:
        lock_file = lock_file.replace('/', '\\')
        terminate_process_using_lock_file(lock_file)
    except Exception as e:
        print('Failed to terminate processes')

# Delete the folder
try:
    shutil.rmtree(folder_path)
    print(f"Folder {folder_path} deleted successfully")
except Exception as e:
    print(f"Failed to delete folder {folder_path}: {e}")
