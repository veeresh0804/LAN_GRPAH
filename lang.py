import os
import sys
import io
import traceback

from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 1. LLM INITIALIZATION
# ==========================================

# Get Gemini API key from environment variable.
# On Render, add:
#
# GEMINI_API_KEY = your_api_key
#

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not configured."
    )


# Initialize Gemini
llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key
)

llm = llm_flash


# ==========================================
# 2. FASTAPI INITIALIZATION
# ==========================================

app = FastAPI(
    title="LAN GRPAH - AI Coding Agent",
    description="LangGraph based Developer and Tester workflow",
    version="1.0.0"
)


# ==========================================
# 3. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):

    messages: List[BaseMessage]

    next_step: Optional[str]

    code: Optional[str]

    report: Optional[str]


# ==========================================
# 4. API REQUEST MODEL
# ==========================================

class TaskRequest(BaseModel):

    task: str


# ==========================================
# 5. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:

    """
    Execute python code and return the
    standard output or error trace.
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


# ==========================================
# TEST CASE GENERATOR
# ==========================================

@tool
def generate_test_cases(task_description: str) -> str:

    """
    Generate specific test scenarios
    for a given coding task.
    """

    prompt = (

        "You are a Senior QA Engineer. "

        "Generate 3 to 5 highly specific "
        "test scenarios for the following "
        "coding task:\n\n"

        f"{task_description}\n\n"

        "Include standard cases and edge cases. "

        "Return them as a numbered list."

    )

    response = llm.invoke(prompt)

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


# ==========================================
# 6. GRAPH NODES
# ==========================================

def task_input_node(state: CrewState):

    """
    Original interactive task input node.

    This is kept for compatibility with
    the original workflow, although Render
    uses the /run-task API instead.
    """

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


# ==========================================
# DEVELOPER NODE
# ==========================================

def real_time_developer(state: CrewState):

    print(
        "\n[Developer] "
        "Writing dynamic code using LLM..."
    )

    # Get latest task
    task = state["messages"][-1].content

    dev_prompt = f"""
You are a Python Developer.

Write a clean Python script to solve this:

{task}

Requirements:
- Return only Python code.
- No explanation.
- No Markdown.
- Do not use ```python.
"""

    # Check LLM
    if llm_flash is None:

        raise ValueError(
            "LLM is not initialized."
        )

    # Call Gemini
    response = llm_flash.invoke(
        dev_prompt
    )

    # Safely parse Gemini response
    content = response.content

    if isinstance(content, list):

        code_parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get(
                    "text",
                    ""
                )

                if text:
                    code_parts.append(text)

            else:

                code_parts.append(
                    str(item)
                )

        code_str = "\n".join(
            code_parts
        )

    else:

        code_str = str(content)

    # Remove Markdown fences if Gemini
    # accidentally returns them
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


# ==========================================
# TESTER NODE
# ==========================================

def real_time_tester(state: CrewState):

    print(
        "\n[Tester] "
        "Generating dynamic tests "
        "and executing code..."
    )

    # Get task
    task = state["messages"][-1].content

    # ======================================
    # Generate test cases
    # ======================================

    test_cases = generate_test_cases.invoke(
        task
    )

    content = test_cases

    if isinstance(content, list):

        cases_parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get(
                    "text",
                    ""
                )

                if text:
                    cases_parts.append(
                        text
                    )

            else:

                cases_parts.append(
                    str(item)
                )

        cases_str = "\n".join(
            cases_parts
        )

    else:

        cases_str = str(content)

    # ======================================
    # Execute generated code
    # ======================================

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # ======================================
    # Compile report
    # ======================================

    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"

        "### TEST SCENARIOS EVALUATED:\n"
        f"{cases_str}"
    )

    return {

        "report": report

    }


# ==========================================
# MANAGER DECISION NODE
# ==========================================

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

    else:

        return {
            "next_step": "task_input"
        }


# ==========================================
# ARCHIVER NODE
# ==========================================

def archiver_node(state: CrewState):

    print(
        "\n[Archiver] "
        "Task stored successfully. "
        "Closing workflow."
    )

    return {

        "next_step": "exit"

    }


# ==========================================
# 7. GRAPH CONSTRUCTION
# ==========================================
# ==========================================
# API WORKFLOW
# ==========================================

api_workflow = StateGraph(CrewState)

api_workflow.add_node(
    "developer",
    real_time_developer
)

api_workflow.add_node(
    "tester",
    real_time_tester
)

api_workflow.add_edge(
    START,
    "developer"
)

api_workflow.add_edge(
    "developer",
    "tester"
)

api_workflow.add_edge(
    "tester",
    END
)

api_app = api_workflow.compile()


# ==========================================
# Add Nodes
# ==========================================

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


# ==========================================
# START → TASK INPUT
# ==========================================

rt_workflow.add_edge(
    START,
    "task_input"
)


# ==========================================
# ROUTING FROM TASK INPUT
# ==========================================

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


# ==========================================
# SEQUENTIAL FLOW
# ==========================================

rt_workflow.add_edge(
    "developer",
    "tester"
)

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# ==========================================
# MANAGER ROUTING
# ==========================================

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


# ==========================================
# ARCHIVER → END
# ==========================================

rt_workflow.add_edge(
    "archiver",
    END
)


# ==========================================
# COMPILE LANGGRAPH
# ==========================================

rt_app = rt_workflow.compile()


print(
    "LangGraph workflow compiled successfully."
)


# ==========================================
# 8. FASTAPI ROUTES
# ==========================================

@app.get("/")
def home():

    return {

        "status": "online",

        "application": "LAN GRPAH",

        "service": "AI Coding Agent",

        "message": "LangGraph API is running."

    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


# ==========================================
# RUN CODING TASK
# ==========================================

@app.post("/run-task")
def run_task(
    request: TaskRequest
):

    task = request.task.strip()

    if not task:

        return {

            "error":
            "Task cannot be empty."

        }

    print("\n" + "=" * 50)

    print(
        "[API] New coding task received:"
    )

    print(task)

    print("=" * 50)


    # ======================================
    # Initial State
    # ======================================

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

        # ==================================
        # Run only Developer → Tester
        # ==================================

        result = api_app.invoke(

            initial_state,

            config={
                "recursion_limit": 50
            },

            # Start from developer
            # because API already supplies
            # the task.
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

        return {

            "status": "error",

            "message": str(e),

            "traceback":
                traceback.format_exc()

        }


# ==========================================
# 9. LOCAL / RENDER SERVER
# ==========================================

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
