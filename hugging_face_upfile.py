from huggingface_hub import HfApi
from huggingface_hub import login
login("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",add_to_git_credential=True)
api = HfApi()

api.upload_folder(
    folder_path=r"E:\NLP\InstaFormer\dataset",
    repo_id="datnguyentien204/InstaFormerDataset",
    repo_type="dataset",multi_commits=True,multi_commits_verbose=True
)