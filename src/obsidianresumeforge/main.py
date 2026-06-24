#!/usr/bin/env python
import os
import sys
from obsidianresumeforge.crew import ObsidianresumeforgeCrew

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _inputs(run_id: str) -> dict:
    return {
        'jd_file_path': os.getenv('JD_FILE_PATH', '/path/to/JDs/YourRole.md'),
        'role_instructions_folder': os.getenv('ROLE_INSTRUCTIONS_FOLDER', os.path.join(_REPO_ROOT, 'knowledge/roles/')),
        'experience_files_folder': os.getenv('EXPERIENCE_FILES_FOLDER', os.path.join(_REPO_ROOT, 'knowledge/experience')),
        'latex_template_path': os.getenv('LATEX_TEMPLATE_PATH', os.path.join(_REPO_ROOT, 'knowledge/template.tex')),
        'vault_path': os.getenv('VAULT_PATH', '/path/to/JobSearch/'),
        'logs_folder': os.getenv('LOGS_FOLDER', '/path/to/JobSearch/'),
        'eval_max_retries': '3',
        'eval_pass_threshold': '0.9',
        'run_id': run_id,
    }

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    ObsidianresumeforgeCrew().crew().kickoff(inputs=_inputs('run_20260624_001'))


def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        ObsidianresumeforgeCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=_inputs('run_20260624_001'))

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ObsidianresumeforgeCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    try:
        ObsidianresumeforgeCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=_inputs('run_20260624_001'))

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
