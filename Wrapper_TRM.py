import os
import json
from typing import Iterator, Dict, Any, List

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.workflow import Workflow, Step, Loop
from agno.workflow.types import StepOutput
from agno.run.workflow import WorkflowRunOutput
from agno.exceptions import StopAgentRun

# --- Setup ---

# Which model are we using?
# (Make sure you've run 'ollama pull gemma3:4b' first)
OLLAMA_MODEL_ID = "gemma3:4b"
CONFIDENCE_THRESHOLD = 0.95 # You can change it max is 1
MAX_ATTEMPTS = 5

# --- The AI Agent ---
# One agent will do all the work
# Give it time to think (increase if needed)
llm = Ollama(id=OLLAMA_MODEL_ID, timeout=180) 

assistant_agent = Agent(
    model=llm,
    description="An assistant that drafts, verifies, and refines answers.",
    markdown=True,
    debug_mode=False, # Set to True for spammy logs
)

# --- The Workflow Steps ---

def draft_step(step_input: Any, session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the LLM to generate the initial draft or refine based on feedback."""
    # Note: step_input type hint changed to Any for flexibility, access via .input
    user_prompt = step_input.input.get("user_prompt") # Get the user's question
    feedback = session_state.get("current_feedback", "")
    # Track our attempts (start at 1)
    attempt = session_state.get("attempt", 1) 
    
    print(f"\n--- Attempt {attempt}: Drafting ---")

    if feedback:
        draft_prompt = f"Refine the previous draft based on this feedback: '{feedback}'. Original prompt: '{user_prompt}'"
    else:
        draft_prompt = f"Draft a comprehensive answer to the following prompt: '{user_prompt}'"
        
    response = assistant_agent.run(draft_prompt)
    current_draft = response.content
    session_state["current_draft"] = current_draft # Save the draft for the verifier
    print(f"Draft (first 100 chars): {current_draft[:100]}...") 
    
    # Bump the attempt counter
    session_state["attempt"] = attempt + 1 
    
    return {"draft": current_draft} # Pass the draft along

def verify_step(step_input: Any, session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the LLM to verify the draft and extract a confidence score."""
    # Note: step_input type hint changed to Any for flexibility, access via .input
    current_draft = session_state.get("current_draft", "No draft available.")
    user_prompt = step_input.input.get("user_prompt") # Get the original question
    attempt = session_state.get("attempt", 1) -1 # Which attempt are we checking?
    if attempt == 0: attempt = 1

    print("--- Verifying ---")

    verify_prompt = f"""
    Analyze the following draft answer for the user prompt.
    Is the draft accurate, complete, and logically sound?
    Provide a confidence score between 0.0 (low confidence, needs major revision) and 1.0 (high confidence, looks good).
    Provide brief feedback for improvement if confidence is below {CONFIDENCE_THRESHOLD}.
    Respond ONLY with a valid JSON object containing "confidence_score" (float) and "feedback" (string).

    USER PROMPT: "{user_prompt}"
    DRAFT ANSWER: "{current_draft}"

    JSON RESPONSE:
    """
    
    try:
        # Ask the AI to check its own work
        raw_response = assistant_agent.run(verify_prompt) 
        
        # Clean up LLM's messy JSON (```json ... ```)
        response_content = raw_response.content.strip()
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
        response_content = response_content.strip()

        verify_data = json.loads(response_content)

        confidence = float(verify_data.get("confidence_score", verify_data.get("confidence", 0.0)))
        feedback = str(verify_data.get("feedback", ""))

        print(f"Confidence: {confidence}, Feedback: {feedback}")
        # Store into session_state
        session_state["current_confidence"] = confidence
        session_state["current_feedback"] = feedback if confidence < CONFIDENCE_THRESHOLD else ""

        # Package results for the loop check
        result = {"confidence": confidence, "feedback": feedback, "attempt": attempt}
        result_json = json.dumps(result)

        # If it's good, mark it as the winner
        if confidence >= CONFIDENCE_THRESHOLD:
            session_state["final_candidate"] = session_state.get("current_draft")

        # Return the results as a JSON string
        return result_json
        
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Error parsing verification JSON response: {e}")
        print(f"Raw response was: {raw_response.content if 'raw_response' in locals() else 'Response not captured'}")
        # If JSON parsing breaks, just try again
        session_state["current_confidence"] = 0.0
        session_state["current_feedback"] = "Verification failed due to parsing error. Please review the draft for general quality."
        # Send failure info back
        return {"confidence": 0.0, "feedback": "Verification failed.", "attempt": attempt}

# --- When Do We Stop? ---

def check_confidence_condition(outputs: List[StepOutput], session_state: Dict[str, Any] = None) -> bool:
    """
    Evaluates if the loop should continue based SOLELY on the outputs
    of the last iteration.
    Returns True to STOP the loop, False to CONTINUE.
    """
    confidence = 0.0
    attempt_completed = 0
    verify_step_output = None

    # Find the verifier's last output
    for output in outputs:
        if output.step_name == "Verify Draft":
            verify_step_output = output
            break

    # Get the confidence and attempt number
    if verify_step_output and verify_step_output.content:
        # The output might be a string or dict, check both
        content = verify_step_output.content
        try:
            if isinstance(content, str):
                parsed = json.loads(content)
            elif isinstance(content, dict):
                parsed = content
            else:
                # Last resort, string-to-JSON
                parsed = json.loads(str(content))
        except Exception:
            parsed = {}

        confidence = float(parsed.get("confidence", parsed.get("confidence_score", 0.0))) if parsed else 0.0
        attempt_completed = int(parsed.get("attempt", 0)) if parsed else 0

    # Prefer the session state if it has info
    if session_state:
        sc = session_state.get("current_confidence")
        sa = session_state.get("attempt")
        if sc is not None:
            try:
                confidence = float(sc)
            except Exception:
                pass
        if sa is not None and attempt_completed == 0:
            try:
                attempt_completed = int(sa) - 1
            except Exception:
                pass

    print(f"Loop check: Attempt completed {attempt_completed}, Confidence {confidence}")

    if confidence >= CONFIDENCE_THRESHOLD:
        print(f"Confidence threshold ({CONFIDENCE_THRESHOLD}) met. Stopping loop.")
        return True # True = STOP looping
    elif attempt_completed >= MAX_ATTEMPTS:
        print(f"Maximum attempts ({MAX_ATTEMPTS}) reached. Stopping loop.")
        return True # True = STOP looping
    else:
        print("Confidence low or verification failed. Continuing loop.")
        return False # False = KEEP looping

# --- Building the Workflow ---
# (Note: The code below redefines draft_step, we will use that new definition)
iterative_loop = Loop(
    name="Iterative Refinement Loop",
    steps=[
        Step(name="Draft", executor=draft_step),
        Step(name="Verify Draft", executor=verify_step),
    ],
    end_condition=check_confidence_condition, # Use our 'check_confidence' decider
)

refinement_workflow = Workflow(
    name="Draft Verify Refine Workflow",
    steps=[iterative_loop],
    # draft_step now handles the attempt counter
    session_state={} # Start fresh
)

# Add one last "polish" step after the loop
def finalize_step(step_input: Any, session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a polished final answer from the confident draft stored in session_state."""
    # Get the winning draft (or the last one)
    draft = session_state.get("final_candidate") or session_state.get("current_draft") or ""
    if not draft:
        return {"final_answer": "No draft available to finalize."}

    print("--- Finalizing into polished answer ---")
    finalize_prompt = f"Polish and produce a final, concise, high-quality answer based on this draft:\n\n{draft}\n\nRespond ONLY with the polished final answer, without any metadata."
    resp = assistant_agent.run(finalize_prompt)
    final_answer = resp.content.strip()

    # Save the final, polished answer
    session_state["final_answer"] = final_answer

    return {"final_answer": final_answer}

# Add the polish step to the main workflow
refinement_workflow.steps = [iterative_loop, Step(name="Finalize", executor=finalize_step)]

# --- (Redefining draft_step) ---
# This version will be used by the loop
def draft_step(step_input: Any, session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the LLM to generate the initial draft or refine based on feedback."""
    user_prompt = step_input.input.get("user_prompt")
    feedback = session_state.get("current_feedback", "")
    # Start at attempt 1 if not already set
    attempt = session_state.get("attempt", 1)

    print(f"\n--- Attempt {attempt}: Drafting ---")

    if feedback:
        draft_prompt = f"Refine the previous draft based on this feedback: '{feedback}'. Original prompt: '{user_prompt}'"
        # Don't reuse old feedback
        session_state["current_feedback"] = ""
    else:
        draft_prompt = f"Draft a comprehensive answer to the following prompt: '{user_prompt}'"

    response = assistant_agent.run(draft_prompt)
    current_draft = response.content
    session_state["current_draft"] = current_draft
    print(f"Draft (first 100 chars): {current_draft[:100]}...")

    # Go to the next attempt
    session_state["attempt"] = attempt + 1

    return {"draft": current_draft}

# --- Let's Run It! ---

if __name__ == "__main__":
    user_question = "You have a standard deck of 52 playing cards. You draw two cards without replacement. If the first card you draw is a King, what is the probability that the second card you draw is also a King? Show your calculation."
    
    print(f"Running workflow for prompt: '{user_question}'")
    
    # Give this run a name
    session_id = "refine_session_001" 

    try:
        # Kick it off!
        final_output: WorkflowRunOutput = refinement_workflow.run(
            input={"user_prompt": user_question},
            session_id=session_id, # Pass in the session name
            stream=False # Don't stream; wait for the full result
        )
        
        print("\n--- Workflow Run Complete ---")

        # Helper to find the final answer
        def _flatten_step_outputs(outputs: List[StepOutput]) -> List[StepOutput]:
            flat = []
            for out in outputs:
                flat.append(out)
                # Loops can nest outputs, so dig in
                nested = getattr(out, "steps", None)
                if nested:
                    flat.extend(_flatten_step_outputs(nested))
            return flat

        # Helper to extract the final data
        def _extract_final_draft_and_loops(run_output: WorkflowRunOutput):
            # First, try the session state
            if hasattr(run_output, "session_state") and run_output.session_state:
                s = run_output.session_state
                return s.get("current_draft", None), max(s.get("attempt", 1) - 1, 1)

            # Next, try the main output content
            if getattr(run_output, "content", None):
                c = run_output.content
                # If it's a dict, check for our keys
                if isinstance(c, dict):
                    return c.get("draft") or c.get("current_draft"), 1
                if isinstance(c, str):
                    try:
                        parsed = json.loads(c)
                        if isinstance(parsed, dict):
                            return parsed.get("draft") or parsed.get("current_draft"), 1
                    except Exception:
                        pass

            # Last, dig through all the step results
            step_results = getattr(run_output, "step_results", None) or getattr(run_output, "steps", None) or getattr(run_output, "step_outputs", None)
            if step_results:
                flat = _flatten_step_outputs(step_results)
                # Find all the draft steps
                draft_steps = [o for o in flat if getattr(o, "step_name", "") == "Draft"]
                verify_steps = [o for o in flat if getattr(o, "step_name", "") == "Verify Draft"]

                loops = max(len(draft_steps), 1)

                # The last verify step is most relevant
                if verify_steps:
                    last_verify = verify_steps[-1]
                # Get the content from the final draft step
                if draft_steps:
                    last_draft_out = draft_steps[-1]
                    content = getattr(last_draft_out, "content", None)
                    if isinstance(content, dict):
                        return content.get("draft") or content.get("current_draft"), loops
                    if isinstance(content, str):
                        return content, loops

            return None, 0

        final_draft, loops_completed = _extract_final_draft_and_loops(final_output)

        # Helper to clean up the final text
        def _normalize_final_text(value: Any) -> str:
            # If it's a dict, get the answer field
            if isinstance(value, dict):
                v = value.get("final_answer") or value.get("draft") or value.get("current_draft")
                if v is None:
                    return json.dumps(value, indent=2)
                value = v

            # If it's a string, try parsing as JSON
            if isinstance(value, str):
                s = value.strip()
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        return parsed.get("final_answer") or parsed.get("draft") or parsed.get("current_draft") or json.dumps(parsed, indent=2)
                    if isinstance(parsed, str):
                        s = parsed
                except Exception:
                    # If not JSON, try parsing as a Python dict
                    try:
                        parsed = None
                        if s.startswith("{") and "'" in s:
                            import ast as _ast
                            parsed = _ast.literal_eval(s)
                        if isinstance(parsed, dict):
                            return parsed.get("final_answer") or parsed.get("draft") or parsed.get("current_draft") or json.dumps(parsed, indent=2)
                    except Exception:
                        pass

                # Fix escaped chars like \n
                s = s.replace('\\n', '\n')
                s = s.replace('\\t', '\t')
                s = s.replace('\\"', '"')
                # Remove extra quotes
                if s.startswith('"') and s.endswith('"'):
                    s = s[1:-1]
                return s.strip()

            # If all else fails, just stringify it
            return str(value)

        final_text = _normalize_final_text(final_draft) if final_draft else "No final draft available in run output."
        if loops_completed == 0:
            loops_completed = 1

        # Show the final result
        sep = "=" * 80
        print(f"\n{sep}\nQUESTION:\n{user_question}\n\nFINAL ANSWER:\n")
        print(final_text)
        print(f"\n{sep}\nCompleted in {loops_completed} loop(s).\n")

    except Exception as e:
        print(f"\n--- Workflow Failed ---")
        print(f"An error occurred: {e}")
        # Dump the session state if it broke
        try:
            current_state = refinement_workflow.get_session_state(session_id=session_id)
            print("\n--- State at time of failure ---")
            print(current_state)
        except Exception as state_e:
            print(f"Could not retrieve session state after failure: {state_e}")
