from huggingface_hub import HfApi
from huggingface_hub import login
login("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",add_to_git_credential=True)
api = HfApi()

api.upload_folder(
    folder_path=r"E:\NLP\InstaFormer\dataset2",
    repo_id="datnguyentien204/InstaFormerDataset",
    repo_type="dataset",multi_commits=True,multi_commits_verbose=True
)