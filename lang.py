import os
import sys
import io
import traceback

from typing import TypedDict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# LANGGRAPH PLAYGROUND
# ============================================================

@app.get("/agent/playground")
def agent_playground():

    return """
<!DOCTYPE html>

<html>

<head>

    <title>LAN GRPAH - Agent Playground</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #0f1117;
            color: #ffffff;
        }

        .container {
            max-width: 1100px;
            margin: 40px auto;
            padding: 20px;
        }

        h1 {
            margin-bottom: 5px;
        }

        .subtitle {
            color: #9ca3af;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            min-height: 150px;
            padding: 15px;
            box-sizing: border-box;
            background: #181b23;
            color: white;
            border: 1px solid #343946;
            border-radius: 8px;
            font-size: 15px;
            resize: vertical;
        }

        button {
            margin-top: 15px;
            padding: 12px 24px;
            border: none;
            border-radius: 7px;
            background: #2563eb;
            color: white;
            font-size: 15px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        button:disabled {
            background: #4b5563;
            cursor: not-allowed;
        }

        .section {
            margin-top: 25px;
        }

        .section h2 {
            font-size: 18px;
        }

        pre {
            background: #181b23;
            border: 1px solid #343946;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
        }

        .status {
            margin-top: 15px;
            color: #9ca3af;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>LAN GRPAH</h1>

    <div class="subtitle">
        LangGraph AI Developer & Tester Playground
    </div>


    <div class="section">

        <h2>Enter Coding Task</h2>

        <textarea
            id="task"
            placeholder="Example: Write a Python program to check whether a number is prime..."
        ></textarea>

        <br>

        <button
            id="runButton"
            onclick="runAgent()"
        >
            Run Agent
        </button>

        <div
            id="status"
            class="status"
        ></div>

    </div>


    <div class="section">

        <h2>Generated Code</h2>

        <pre id="code">Waiting for agent...</pre>

    </div>


    <div class="section">

        <h2>Agent Report</h2>

        <pre id="report">Waiting for agent...</pre>

    </div>

</div>


<script>

async function runAgent() {

    const task =
        document.getElementById("task").value.trim();

    const button =
        document.getElementById("runButton");

    const status =
        document.getElementById("status");

    const code =
        document.getElementById("code");

    const report =
        document.getElementById("report");


    if (!task) {

        alert("Please enter a coding task.");

        return;
    }


    button.disabled = true;

    button.innerText = "Running Agent...";

    status.innerText =
        "Developer → Tester → Report";


    code.innerText =
        "Generating code...";

    report.innerText =
        "Running LangGraph...";


    try {

        const response = await fetch(
            "/run-task",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    task: task
                })
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Agent execution failed."
            );
        }


        code.innerText =
            data.code || "No code generated.";


        report.innerText =
            data.report || "No report generated.";


        status.innerText =
            "Agent completed successfully.";

    }

    catch (error) {

        status.innerText =
            "Agent execution failed.";

        code.innerText =
            "Error";

        report.innerText =
            error.message;

    }

    finally {

        button.disabled = false;

        button.innerText =
            "Run Agent";

    }

}

</script>


</body>

</html>
"""

# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not configured."
    )

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key
)

llm = llm_flash


# ============================================================
# 2. FASTAPI
# ============================================================

app = FastAPI(
    title="LAN GRPAH",
    description="LangGraph AI Developer and Tester",
    version="1.0.0"
)


# ============================================================
# 3. STATE DEFINITION
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 4. API REQUEST MODEL
# ============================================================

class TaskRequest(BaseModel):
    task: str


# ============================================================
# 5. TOOLS
# ============================================================

@tool
def _python_code(code: str) -> str:
    """
    Execute Python code and return standard output
    or an error trace.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = new_stdout.getvalue()

    except Exception:
        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


# ============================================================
# TEST CASE GENERATOR
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate specific test scenarios for a coding task.
    """

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for the following coding task:

{task_description}

Include:
- Standard cases
- Edge cases
- Boundary cases
- Invalid input cases where relevant

Return them as a numbered list.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):
                parts.append(
                    item.get("text", "")
                )
            else:
                parts.append(str(item))

        return "\n".join(
            p for p in parts if p
        )

    return str(content)


# ============================================================
# 6. TASK INPUT NODE
# ============================================================

def task_input_node(state: CrewState):

    print("\n" + "=" * 50)

    print("--- NEW TASK INITIALIZATION ---")

    user_task = input(
        "Enter the coding task "
        "(or type 'exit' to quit): "
    ).strip()

    if user_task.lower() == "exit":

        return {
            "next_step": "exit"
        }

    return {
        "messages": [
            HumanMessage(
                content=user_task
            )
        ],
        "next_step": "developer"
    }


# ============================================================
# 7. DEVELOPER NODE
# ============================================================

def real_time_developer(state: CrewState):

    print(
        "\n[Developer] "
        "Writing dynamic code using LLM..."
    )

    task = state["messages"][-1].content

    dev_prompt = f"""
You are a Senior Python Developer.

Write a clean Python script to solve this coding task:

{task}

Requirements:

- Return ONLY Python code.
- No explanation.
- No Markdown.
- Do not include ```python.
- The code must be executable.
"""

    response = llm_flash.invoke(
        dev_prompt
    )

    content = response.content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get(
                    "text",
                    ""
                )

                if text:
                    parts.append(text)

            else:

                parts.append(
                    str(item)
                )

        code_str = "\n".join(parts)

    else:

        code_str = str(content)

    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print(code_str)

    return {
        "code": code_str
    }


# ============================================================
# 8. TESTER NODE
# ============================================================

def real_time_tester(state: CrewState):

    print(
        "\n[Tester] "
        "Generating dynamic tests "
        "and executing code..."
    )

    task = state["messages"][-1].content

    # Generate tests
    test_cases = generate_test_cases.invoke(
        task
    )

    cases_str = str(test_cases)

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # Compile report
    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED:\n"
        f"{cases_str}"
    )

    return {
        "report": report
    }


# ============================================================
# 9. MANAGER NODE
# ============================================================

def manager_decision_node(state: CrewState):

    print("\n" + "=" * 50)

    print(
        "--- MANAGER DASHBOARD : "
        "TEST REPORT ---"
    )

    print(
        state.get(
            "report",
            "No report available."
        )
    )

    print("=" * 50)

    user_input = input(
        "\nCommand (store / another): "
    ).lower().strip()

    if user_input == "store":

        return {
            "next_step": "archiver"
        }

    return {
        "next_step": "task_input"
    }


# ============================================================
# 10. ARCHIVER NODE
# ============================================================

def archiver_node(state: CrewState):

    print(
        "\n[Archiver] "
        "Task stored successfully. "
        "Closing workflow."
    )

    return {
        "next_step": "exit"
    }


# ============================================================
# 11. ORIGINAL INTERACTIVE WORKFLOW
# ============================================================

rt_workflow = StateGraph(
    CrewState
)

rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# START → task_input

rt_workflow.add_edge(
    START,
    "task_input"
)


def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":
        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# developer → tester

rt_workflow.add_edge(
    "developer",
    "tester"
)


# tester → manager

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


def route_from_decision(
    state: CrewState
):

    if state.get(
        "next_step"
    ) == "archiver":

        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# archiver → END

rt_workflow.add_edge(
    "archiver",
    END
)


# Compile original workflow

rt_app = rt_workflow.compile()


# ============================================================
# 12. API WORKFLOW
# ============================================================

# This workflow is separate from the
# interactive terminal workflow.

api_workflow = StateGraph(
    CrewState
)

api_workflow.add_node(
    "developer",
    real_time_developer
)

api_workflow.add_node(
    "tester",
    real_time_tester
)


# API START → developer

api_workflow.add_edge(
    START,
    "developer"
)


# developer → tester

api_workflow.add_edge(
    "developer",
    "tester"
)


# tester → END

api_workflow.add_edge(
    "tester",
    END
)


# Compile API workflow

api_app = api_workflow.compile()


print(
    "LangGraph workflows compiled successfully."
)


# ============================================================
# 13. FASTAPI ROUTES
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "application": "LAN GRPAH",
        "service": "AI Coding Agent",
        "message": "LangGraph API is running."
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# RUN CODING TASK
# ============================================================

@app.post("/run-task")
def run_task(
    request: TaskRequest
):

    task = request.task.strip()

    if not task:

        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty."
        )

    print("\n" + "=" * 50)

    print(
        "[API] New coding task received:"
    )

    print(task)

    print("=" * 50)


    # Initial LangGraph state

    initial_state: CrewState = {

        "messages": [
            HumanMessage(
                content=task
            )
        ],

        "next_step": "developer",

        "code": None,

        "report": None
    }


    try:

        # Run API-specific workflow

        result = api_app.invoke(
            initial_state,
            config={
                "recursion_limit": 50
            }
        )


        return {
            "status": "success",
            "task": task,
            "code": result.get(
                "code",
                ""
            ),
            "report": result.get(
                "report",
                ""
            )
        }


    except Exception as e:

        print(
            "\nERROR:\n"
            + traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 14. SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        "lang:app",
        host="0.0.0.0",
        port=port
    )
