"""Hugging Face Space for the Unit 4 final assignment, wired to the gaia_agent package.

Two ways to submit:
  1. "Run Evaluation & Submit All Answers": the course template flow. Fetches the questions, runs
     gaia_agent.solve on each one (with a per-question timeout), then submits everything.
  2. "Submit cached answers": submits runs/answers.jsonl (produced locally by `python run.py` and
     committed with the Space), so the Space does not need to re-run all 20 questions.
"""

import os

import gradio as gr
import pandas as pd
import requests

from gaia_agent.config import API_URL, get_run_id, provider_info
from gaia_agent.run import ANSWERS_PATH, append_record, run_with_timeout
from gaia_agent.submit import build_payload

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = API_URL
PER_QUESTION_TIMEOUT = float(os.getenv("GAIA_TIMEOUT", "420"))


# --- Agent Definition ---
class GaiaAgent:
    """Callable wrapper around gaia_agent.solve, keeping the template's agent(question) shape."""

    def __init__(self):
        from gaia_agent import solve

        self._solve = solve
        print(f"GaiaAgent initialized. Provider: {provider_info()}")

    def __call__(self, question: str, task_id: str | None = None, file_name: str | None = None) -> dict:
        item = {"task_id": task_id, "question": question, "file_name": file_name}
        return run_with_timeout(self._solve, item, PER_QUESTION_TIMEOUT)


def _agent_code() -> str:
    space_id = os.getenv("SPACE_ID") or "bacool/gaia-agent"
    return f"https://huggingface.co/spaces/{space_id}/tree/main"


def _submit(submission_data: dict, results_log: list) -> tuple[str, pd.DataFrame]:
    submit_url = f"{DEFAULT_API_URL}/submit"
    print(f"Submitting {len(submission_data['answers'])} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return final_status, pd.DataFrame(results_log)
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
    print(status_message)
    return status_message, pd.DataFrame(results_log)


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetches all questions, runs the GaiaAgent on them, submits all answers, and displays the results."""
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    questions_url = f"{DEFAULT_API_URL}/questions"

    # 1. Instantiate Agent
    try:
        agent = GaiaAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    agent_code = _agent_code()
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    info = provider_info()
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        res = agent(question_text, task_id=task_id, file_name=item.get("file_name") or None)
        submitted_answer = res.get("answer", "")
        try:
            append_record(
                {"run_id": res.get("run_id") or get_run_id(), "task_id": task_id, "question": question_text,
                 "answer": submitted_answer, "raw": res.get("raw"), "steps": res.get("steps"),
                 "error": res.get("error"), "trace_path": res.get("trace_path"), "provider_info": info},
                ANSWERS_PATH,
            )
        except Exception as e:  # the Space filesystem may be read-only; caching is best-effort
            print(f"Could not cache answer: {e}")
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append(
            {"Task ID": task_id, "Question": question_text,
             "Submitted Answer": submitted_answer if not res.get("error") else f"AGENT ERROR: {res['error']}"}
        )

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission + 5. Submit
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    print(f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'...")
    return _submit(submission_data, results_log)


def submit_cached(profile: gr.OAuthProfile | None):
    """Submit the answers cached in runs/answers.jsonl without re-running the agent."""
    if not profile:
        return "Please Login to Hugging Face with the button.", None
    username = f"{profile.username}"
    if not ANSWERS_PATH.is_file():
        return f"No cached answers found at {ANSWERS_PATH.relative_to(ANSWERS_PATH.parent.parent)}.", None
    payload = build_payload(ANSWERS_PATH, username=username, agent_code=_agent_code())
    if not payload["answers"]:
        return "The answers cache is empty.", None
    results_log = [{"Task ID": a["task_id"], "Submitted Answer": a["submitted_answer"]} for a in payload["answers"]]
    return _submit(payload, results_log)


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1. Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        2. Click **Run Evaluation & Submit All Answers** to fetch the questions, run the agent on every one
           (this takes a long time), and submit.
        3. Or click **Submit cached answers** to submit the answers the agent already produced locally
           (`runs/answers.jsonl`), without re-running anything.

        The agent is a smolagents `CodeAgent` with web, Wikipedia, YouTube, audio, image and file tools.
        Model provider: set `GAIA_PROVIDER` and the matching API key as Space secrets.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")
    cached_button = gr.Button("Submit cached answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])
    cached_button.click(fn=submit_cached, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-" * (60 + len(" App Starting ")) + "\n")
    print("Launching Gradio Interface for GAIA Agent Evaluation...")
    demo.launch(debug=True, share=False)
