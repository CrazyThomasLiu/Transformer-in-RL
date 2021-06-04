import os

algo = "trdqn"
exp_folder="logs/"
log_path = os.path.join(exp_folder, algo)
env="BreakoutNoFrameskip-v4"
dirs = [
    os.path.join(log_path, folder)
    for folder in os.listdir(log_path)
    if (env in folder and os.path.isdir(os.path.join(log_path, folder)))
]