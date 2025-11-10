import os, json


def collect_experiments(results_dir='results'):

    cwd = os.getcwd()
    os.chdir("../../")
    print(os.getcwd())

    experiments = []
    for subdir in os.listdir(results_dir):

        path = os.path.join(results_dir, subdir)
        if not os.path.isdir(path):
            continue

        # Parse folder name: {seed}_{timestamp}_{short_name}
        parts = subdir.split('_', 2)
        if len(parts) != 3:
            continue

        seed, timestamp, short_name = parts
        
        metrics_path = os.path.join(path, 'evaluation.json')
        config_path = os.path.join(path, 'config.json')

        if not os.path.exists(metrics_path) or not os.path.exists(config_path):
            continue

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            rmse = metrics.get('rmse(valid)')
            if rmse is None:
                continue  # Skip if RMSE not found
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Append to list
        experiments.append({
            'seed': seed,
            'short_name': short_name,
            "timestamp": timestamp,
            'rmse': rmse,
            'config': config
        })

    os.chdir(cwd)

    return experiments
