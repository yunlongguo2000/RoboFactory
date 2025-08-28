import os
import sys
import zipfile
import requests
import tempfile
import math
import time
import json
import shutil
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)


def get_uploaded_chunks(backend_url, submission_id, api_key, verify_ssl=False):
    url = backend_url.rstrip('/') + '/api/submit/uploaded_chunks'
    params = {'submission_id': submission_id, 'api_key': api_key}
    try:
        resp = requests.get(url, params=params, timeout=30, verify=verify_ssl)
        resp.raise_for_status()
        return set(resp.json().get('uploaded_chunks', []))
    except Exception as e:
        print(f"Warning: Could not query uploaded chunks: {e}")
        return set()


def validate_file_path(path, path_type):
    """Validate that a file path exists and is a file."""
    if not os.path.exists(path):
        print(f"Error: {path_type} does not exist: {path}")
        return False
    if not os.path.isfile(path):
        print(f"Error: {path_type} is not a file: {path}")
        return False
    print(f"{path_type} validated: {path}")
    return True


def validate_folder_path(path, path_type):
    """Validate that a folder path exists and is a directory."""
    if not os.path.exists(path):
        print(f"Error: {path_type} does not exist: {path}")
        return False
    if not os.path.isdir(path):
        print(f"Error: {path_type} is not a directory: {path}")
        return False
    print(f"{path_type} validated: {path}")
    return True


def create_submission_structure(checkpoint_file, config_file, policy_folder, temp_dir):
    """Create the proper submission folder structure."""
    submission_dir = os.path.join(temp_dir, "submission")
    os.makedirs(submission_dir, exist_ok=True)

    # Create checkpoints directory and copy checkpoint file
    checkpoints_dir = os.path.join(submission_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_name = os.path.basename(checkpoint_file)
    shutil.copy2(checkpoint_file, os.path.join(checkpoints_dir, checkpoint_name))

    # Create configs directory and copy config file
    configs_dir = os.path.join(submission_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    # Create table subdirectory for config
    table_dir = os.path.join(configs_dir, "table")
    os.makedirs(table_dir, exist_ok=True)
    config_name = os.path.basename(config_file)
    shutil.copy2(config_file, os.path.join(table_dir, config_name))

    # Copy custom policy folder
    custom_policy_dir = os.path.join(submission_dir, "custom_policy")
    shutil.copytree(policy_folder, custom_policy_dir, dirs_exist_ok=True)

    print(f"Created submission structure in: {submission_dir}")
    print("Submission folder contents:")
    for root, dirs, files in os.walk(submission_dir):
        level = root.replace(submission_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

    return submission_dir


def upload_in_chunks(zip_path, backend_url, submission_id, api_key, verify_ssl=False, chunk_size=5 * 1024 * 1024,
                     max_retries=5):
    file_size = os.path.getsize(zip_path)
    total_chunks = math.ceil(file_size / chunk_size)
    uploaded_chunks = get_uploaded_chunks(backend_url, submission_id, api_key, verify_ssl)
    print(f"Already uploaded chunks: {sorted(uploaded_chunks)}")
    with open(zip_path, 'rb') as f:
        for chunk_index in range(total_chunks):
            if chunk_index in uploaded_chunks:
                print(f"Chunk {chunk_index + 1}/{total_chunks} already uploaded, skipping.")
                continue
            f.seek(chunk_index * chunk_size)
            chunk_data = f.read(chunk_size)
            for attempt in range(max_retries):
                try:
                    files = {'file': (os.path.basename(zip_path), chunk_data, 'application/zip')}
                    data = {
                        'submission_id': submission_id,
                        'api_key': api_key,
                        'chunk_index': chunk_index,
                        'total_chunks': total_chunks
                    }
                    upload_url = backend_url.rstrip('/') + '/api/submit/upload_chunk'
                    resp = requests.post(upload_url, files=files, data=data, timeout=60, verify=verify_ssl)
                    resp.raise_for_status()
                    res_json = resp.json()
                    print(f"Chunk {chunk_index + 1}/{total_chunks} upload response:", res_json)
                    if res_json.get('merged'):
                        print('All chunks uploaded and merged.')
                    break
                except Exception as e:
                    # Check if the error is an HTTP 400 (already evaluated)
                    if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                        try:
                            err_json = e.response.json()
                            if 'already evaluated' in str(err_json.get('error', '')).lower():
                                print(
                                    f"Error uploading chunk {chunk_index + 1}: Submission already evaluated (HTTP 400).")
                                print("Aborting upload.")
                                sys.exit(1)
                        except Exception:
                            pass
                    print(f"Error uploading chunk {chunk_index + 1}, attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        print('Max retries reached. Aborting.')
                        sys.exit(1)
                    time.sleep(2)


def main():
    if len(sys.argv) != 6:
        print(
            'Usage: python python_submit.py <submission_id> <api_key> <checkpoint_file> <config_file> <policy_folder>')
        print(
            'Example: python python_submit.py sub123 key123 ./checkpoints/last.ckpt ./configs/table/place_food.yaml ./custom_policy')
        print('Arguments:')
        print('  checkpoint_file: Path to checkpoint file')
        print('  config_file: Path to config file')
        print('  policy_folder: Path to custom policy folder')
        sys.exit(1)

    submission_id, api_key, checkpoint_file, config_file, policy_folder = sys.argv[1:6]

    # Default backend URL - can be overridden by environment variable
    backend_url = os.environ.get('BACKEND_URL', 'https://mygo.iostream.site')

    # SSL verification setting - can be overridden by environment variable
    verify_ssl = os.environ.get('VERIFY_SSL', 'false').lower() == 'true'

    # Validate all input paths
    print("Validating input paths...")
    if not validate_file_path(checkpoint_file, "Checkpoint file"):
        sys.exit(1)
    if not validate_file_path(config_file, "Config file"):
        sys.exit(1)
    if not validate_folder_path(policy_folder, "Policy folder"):
        sys.exit(1)

    # Create temporary directory for submission structure
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating submission folder structure...")
        submission_dir = create_submission_structure(checkpoint_file, config_file, policy_folder, temp_dir)

        # Zip the submission folder
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            zip_folder(submission_dir, tmp_zip.name)
            zip_path = tmp_zip.name
            print(f"Created zip file: {zip_path}")

        # Upload the zip file
        print("Uploading submission...")
        upload_in_chunks(zip_path, backend_url, submission_id, api_key, verify_ssl)

        # Trigger evaluation
        print("Triggering evaluation...")
        eval_url = backend_url.rstrip('/') + '/api/submit/eval'
        eval_data = {
            'submission_id': submission_id,
            'api_key': api_key
        }
        try:
            resp = requests.post(eval_url, json=eval_data, verify=verify_ssl)
            try:
                resp_json = resp.json()
            except Exception:
                resp_json = None
            else:
                print('Eval response:', resp_json if resp_json is not None else resp.text)
        except Exception as e:
            print(f"Error during evaluation request: {e}")

        # Clean up zip file
        os.remove(zip_path)
        print("Upload completed successfully!")


if __name__ == '__main__':
    main()
