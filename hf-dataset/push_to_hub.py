from huggingface_hub import login, upload_folder


login()


upload_folder(folder_path=".", repo_id="NLPC-UOM/anonymized-sinhala-letter-corpus", repo_type="dataset")