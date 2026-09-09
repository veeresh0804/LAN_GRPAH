import os
import sys
import io
import traceback

from typing import TypedDict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LAN GRPAH - AI Coding Agent",
    description="LangGraph based AI Developer and Tester",
    version="1.0.0"
)


# ============================================================
# 2. GEMINI INITIALIZATION
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
def run_python_code(code: str) -> str:
    """
    Execute Python code and return standard output
    or an error trace.
    """

    if not isinstance(code, str):
        code = str(code)

    # Remove Markdown code fences
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

1. Standard cases
2. Edge cases
3. Boundary cases
4. Invalid input cases where relevant

Return them as a numbered list.
"""

    response = llm.invoke(prompt)

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

        return "\n".join(parts)

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

    # Get latest task
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
- Include print statements to demonstrate the result.
"""

    # Call Gemini
    response = llm_flash.invoke(
        dev_prompt
    )

    # Gemini response
    content = response.content

    # Handle normal string response
    if isinstance(content, str):

        code_str = content

    # Handle Gemini structured content
    elif isinstance(content, list):

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

    # Remove accidental Markdown
    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print("\nGenerated Code:")
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

    # Get task
    task = state["messages"][-1].content

    # --------------------------------------------------------
    # Generate test cases
    # --------------------------------------------------------

    test_cases = generate_test_cases.invoke(
        task
    )

    cases_str = str(test_cases)

    # --------------------------------------------------------
    # Execute generated code
    # --------------------------------------------------------

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # --------------------------------------------------------
    # Compile report
    # --------------------------------------------------------

    report = (
        "### EXECUTION OUTPUT:\n\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED:\n\n"
        f"{cases_str}"
    )

    print("\nTester Report:")
    print(report)

    return {

        "report": report

    }


# ============================================================
# 9. MANAGER DECISION NODE
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
# 11. ORIGINAL INTERACTIVE LANGGRAPH
# ============================================================

rt_workflow = StateGraph(
    CrewState
)


# Add nodes

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


# ------------------------------------------------------------
# START → task_input
# ------------------------------------------------------------

rt_workflow.add_edge(
    START,
    "task_input"
)


# ------------------------------------------------------------
# Route after task input
# ------------------------------------------------------------

def route_from_input(state: CrewState):

    if state.get(
        "next_step"
    ) == "exit":

        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# ------------------------------------------------------------
# Developer → Tester
# ------------------------------------------------------------

rt_workflow.add_edge(
    "developer",
    "tester"
)


# ------------------------------------------------------------
# Tester → Manager
# ------------------------------------------------------------

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# ------------------------------------------------------------
# Manager routing
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Archiver → END
# ------------------------------------------------------------

rt_workflow.add_edge(
    "archiver",
    END
)


# Compile original workflow

rt_app = rt_workflow.compile()


# ============================================================
# 12. API LANGGRAPH WORKFLOW
# ============================================================

# The API already receives the task from the browser.
#
# Therefore it must NOT go through task_input_node(),
# because task_input_node() uses input().
#
# API workflow:
#
# START
#   ↓
# Developer
#   ↓
# Tester
#   ↓
# END
#

api_workflow = StateGraph(
    CrewState
)


# Add API nodes

api_workflow.add_node(
    "developer",
    real_time_developer
)

api_workflow.add_node(
    "tester",
    real_time_tester
)


# ------------------------------------------------------------
# API START → Developer
# ------------------------------------------------------------

api_workflow.add_edge(
    START,
    "developer"
)


# ------------------------------------------------------------
# Developer → Tester
# ------------------------------------------------------------

api_workflow.add_edge(
    "developer",
    "tester"
)


# ------------------------------------------------------------
# Tester → END
# ------------------------------------------------------------

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
# 13. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {

        "status": "online",

        "application": "LAN GRPAH",

        "service": "AI Coding Agent",

        "message":
            "LangGraph API is running.",

        "playground":
            "/agent/playground",

        "documentation":
            "/docs"

    }


# ============================================================
# 14. HEALTH ROUTE
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "service": "LAN GRPAH"

    }


# ============================================================
# 15. AGENT PLAYGROUND
# ============================================================

@app.get(
    "/agent/playground",
    response_class=HTMLResponse
)
def agent_playground():

    return """
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>
        LAN GRPAH - Agent Playground
    </title>


    <style>

        * {
            box-sizing: border-box;
        }


        body {

            margin: 0;

            min-height: 100vh;

            font-family:
                Arial,
                Helvetica,
                sans-serif;

            background:
                #0f1117;

            color:
                #ffffff;
        }


        .container {

            width: 90%;

            max-width: 1200px;

            margin:
                0 auto;

            padding:
                40px 0;
        }


        .header {

            margin-bottom:
                30px;
        }


        .header h1 {

            margin:
                0 0 8px 0;

            font-size:
                32px;
        }


        .header p {

            margin:
                0;

            color:
                #9ca3af;

            font-size:
                15px;
        }


        .card {

            background:
                #181b23;

            border:
                1px solid #303541;

            border-radius:
                12px;

            padding:
                22px;

            margin-bottom:
                20px;
        }


        .card h2 {

            margin-top:
                0;

            font-size:
                18px;
        }


        textarea {

            width:
                100%;

            min-height:
                150px;

            resize:
                vertical;

            padding:
                15px;

            border-radius:
                8px;

            border:
                1px solid #343946;

            background:
                #10131a;

            color:
                #ffffff;

            font-size:
                15px;

            outline:
                none;
        }


        textarea:focus {

            border-color:
                #4f7cff;
        }


        button {

            margin-top:
                15px;

            padding:
                12px 25px;

            border:
                none;

            border-radius:
                8px;

            background:
                #2563eb;

            color:
                #ffffff;

            font-size:
                15px;

            cursor:
                pointer;
        }


        button:hover {

            background:
                #1d4ed8;
        }


        button:disabled {

            background:
                #4b5563;

            cursor:
                not-allowed;
        }


        .status {

            margin-top:
                15px;

            color:
                #9ca3af;

            font-size:
                14px;
        }


        pre {

            margin:
                0;

            padding:
                18px;

            background:
                #10131a;

            border:
                1px solid #303541;

            border-radius:
                8px;

            overflow-x:
                auto;

            white-space:
                pre-wrap;

            word-break:
                break-word;

            color:
                #e5e7eb;

            font-family:
                Consolas,
                Monaco,
                monospace;

            font-size:
                14px;

            line-height:
                1.6;
        }


        .flow {

            display:
                flex;

            align-items:
                center;

            justify-content:
                center;

            gap:
                10px;

            flex-wrap:
                wrap;

            margin-top:
                10px;
        }


        .node {

            padding:
                10px 18px;

            border:
                1px solid #3b4250;

            border-radius:
                8px;

            background:
                #20242e;

            font-size:
                14px;
        }


        .arrow {

            color:
                #6b7280;

            font-size:
                18px;
        }


        @media (max-width: 700px) {

            .container {

                width:
                    94%;

                padding-top:
                    25px;
            }


            .header h1 {

                font-size:
                    26px;
            }

        }

    </style>

</head>


<body>


<div class="container">


    <div class="header">

        <h1>
            LAN GRPAH
        </h1>

        <p>
            LangGraph AI Developer & Tester Playground
        </p>

    </div>


    <!-- WORKFLOW -->

    <div class="card">

        <h2>
            Agent Workflow
        </h2>

        <div class="flow">

            <div class="node">
                Task
            </div>

            <div class="arrow">
                →
            </div>

            <div class="node">
                Developer
            </div>

            <div class="arrow">
                →
            </div>

            <div class="node">
                Tester
            </div>

            <div class="arrow">
                →
            </div>

            <div class="node">
                Report
            </div>

        </div>

    </div>


    <!-- TASK -->

    <div class="card">

        <h2>
            Coding Task
        </h2>


        <textarea
            id="task"
            placeholder="Example: Write a Python program to check whether a number is prime."
        ></textarea>


        <button
            id="runButton"
            onclick="runAgent()"
        >
            Run Agent
        </button>


        <div
            id="status"
            class="status"
        >
            Ready
        </div>

    </div>


    <!-- GENERATED CODE -->

    <div class="card">

        <h2>
            Generated Code
        </h2>


        <pre id="code">Waiting for agent...</pre>

    </div>


    <!-- REPORT -->

    <div class="card">

        <h2>
            Tester Report
        </h2>


        <pre id="report">Waiting for agent...</pre>

    </div>


</div>


<script>

async function runAgent() {

    const taskElement =
        document.getElementById("task");

    const button =
        document.getElementById("runButton");

    const status =
        document.getElementById("status");

    const code =
        document.getElementById("code");

    const report =
        document.getElementById("report");


    const task =
        taskElement.value.trim();


    if (!task) {

        alert(
            "Please enter a coding task."
        );

        return;
    }


    // Disable button

    button.disabled = true;

    button.innerText =
        "Running Agent...";


    status.innerText =
        "Developer agent is working...";


    code.innerText =
        "Generating Python code...";


    report.innerText =
        "Waiting for Tester...";


    try {

        const response = await fetch(
            "/run-task",
            {

                method:
                    "POST",

                headers:
                    {
                        "Content-Type":
                            "application/json"
                    },

                body:
                    JSON.stringify({
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


        // Show generated code

        code.innerText =
            data.code ||
            "No code generated.";


        // Show report

        report.innerText =
            data.report ||
            "No report generated.";


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
# 16. RUN TASK API
# ============================================================

@app.post("/run-task")
def run_task(
    request: TaskRequest
):

    task = request.task.strip()


    # Validate task

    if not task:

        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty."
        )


    print("\n" + "=" * 60)

    print(
        "[API] New coding task received:"
    )

    print(task)

    print("=" * 60)


    # ========================================================
    # Initial State
    # ========================================================

    initial_state: CrewState = {

        "messages": [

            HumanMessage(
                content=task
            )

        ],

        "next_step":
            "developer",

        "code":
            None,

        "report":
            None

    }


    try:

        # ====================================================
        # Run API LangGraph
        # ====================================================

        result = api_app.invoke(

            initial_state,

            config={
                "recursion_limit": 50
            }

        )


        # ====================================================
        # Return Result
        # ====================================================

        return {

            "status":
                "success",

            "task":
                task,

            "code":
                result.get(
                    "code",
                    ""
                ),

            "report":
                result.get(
                    "report",
                    ""
                )

        }


    except Exception as e:

        print(
            "\nERROR:\n"
        )

        print(
            traceback.format_exc()
        )


        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# 17. LOCAL SERVER
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
