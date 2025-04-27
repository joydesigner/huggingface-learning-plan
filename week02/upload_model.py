from huggingface_hub import HfApi, HfFolder

# upload model to huggingface hub
api = HfApi()
api.upload_folder(
    folder_path="./results/checkpoint-6250",
    path_in_repo=".",
    repo_id="jzen385/distilbert-base-uncased-finetuned-imdb",
    repo_type="model",
    commit_message="upload model",
    token=HfFolder.get_token(),
)