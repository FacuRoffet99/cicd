import os
from utils.wandb import promote_run_by_id

assert os.environ['WANDB_API_KEY'], 'You must set the WANDB_API_KEY environment variable'

if __name__ == "__main__":
    entity, project, collection = 'FacuRoffet99', 'pytorch-intro', 'MNIST Classifier'
    promote_run_by_id(entity, project, collection, os.environ['RUN_ID'], os.environ['REGISTRY_TAG'])