import yaml
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

with open(current_folder+'/config_workload.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

dataset_cfg = cfg['dataset']

ptb_path = dataset_cfg['ptb_path']

workload_cfg = cfg['workload']

workload_size = workload_cfg['workload_size']
cv_light_ratio = workload_cfg['cv_light_ratio']
cv_med_ratio = workload_cfg['cv_med_ratio']
cv_heavy_ratio = workload_cfg['cv_heavy_ratio']
nlp_light_ratio = workload_cfg['nlp_light_ratio']
nlp_med_ratio = workload_cfg['nlp_med_ratio']
nlp_heavy_ratio = workload_cfg['nlp_heavy_ratio']

objective_cfg = cfg['objective']

convergence_ratio = objective_cfg['convergence_ratio']
accuracy_ratio = objective_cfg['accuracy_ratio']
runtime_ratio = objective_cfg['runtime_ratio']

random_seed = cfg['random_seed']