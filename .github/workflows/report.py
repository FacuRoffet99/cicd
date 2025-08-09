import os
from utils.wandb import get_run_by_unique_tag, get_run_by_id, create_comparison_report

assert os.environ['WANDB_API_KEY'], 'You must set the WANDB_API_KEY environment variable'

if __name__ == "__main__":
    entity, project = 'FacuRoffet99', 'pytorch-intro'

    base_run = get_run_by_unique_tag(entity, project, os.environ['RUN_TAG'])
    new_run = get_run_by_id(entity, project, os.environ['RUN_ID'])
    report_url = create_comparison_report(entity, project, ['RUN_TAG'], base_run, new_run)