import os
import wandb
from urllib.parse import urlencode
import wandb_workspaces.reports.v2 as wr

def get_run_by_unique_tag(entity: str, project: str, tag: str) -> wandb.apis.public.runs.Run:
    """
    Retrieves a run from a tag (which is not used in any other run).
    """
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}', {"tags": {"$in": [tag]}})

    assert runs, f"No '{tag}' run found. Please ensure one run is tagged as '{tag}'."
    assert len(runs) == 1, f"Multiple runs found: {len(runs)}. Please ensure only one run is tagged as '{tag}'."

    return runs[0]

def get_run_by_id(entity: str, project: str, run_id: str) -> wandb.apis.public.runs.Run:
    """
    Retrieves a run from a project using its ID.
    """
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_id}')
    
    assert run, f"No run found with ID '{run_id}'. Please ensure the run ID is correct."
    return run

def create_comparison_report(entity: str, project: str, tag:str, 
                             base_run: wandb.apis.public.runs.Run, 
                             new_run: wandb.apis.public.runs.Run) -> str:
    """
    Creates a WandB report comparing two runs and return the report URL
    """
    report = wr.Report(entity=entity, project=project,
                       title='Run comparison',
                       width='fluid',
                       description=f"New run: {new_run.name}\n'{tag.capitalize()}' run: {base_run.name}") 
    blocks = [
        wr.PanelGrid(
            runsets=[wr.Runset(entity, project, "Run comparison", filters=f"name in ['{new_run.id}', '{base_run.id}']")],
            panels=[wr.RunComparer(diff_only='split', layout={'x': 0, 'y': 0, 'w': 24, 'h': 15})]
        )
    ]
    report.blocks = blocks
    report.save()

    if os.getenv('CI'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            print(f'REPORT_URL={report.url}', file=f)
    return report.url

def promote_run_by_id(entity: str, project: str, collection: str, run_id: str, tag: str) -> None:
    """
    Promotes a run to the Model Registry from a project using its ID and gives it a tag.
    """
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_id}')
    registry_path = f'{entity}/model-registry/{collection}'

    artifacts = [a for a in run.logged_artifacts() if a.type == 'model']
    assert len(artifacts) >= 1, 'No artifacts of type model found.'
    artifacts[-1].link(registry_path, aliases=[tag])   

    versions = api.artifacts('model', registry_path)
    latest_model = versions[0]
    query = urlencode({'selectionPath': registry_path, 'version': latest_model.version})
    registry_url = f'https://wandb.ai/{latest_model.entity}/registry/model?{query}'

    if os.getenv('CI'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            print(f'REGISTRY_URL={registry_url}', file=f)

    return registry_url
    