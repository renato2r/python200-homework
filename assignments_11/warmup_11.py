# assignments_11/warmup_11.py

"""
====================================================================
Week 11 Warmup: Prefect Orchestration & Production Patterns
====================================================================
"""

# --- Prefect Orchestration ---
# Prefect Question 1

# What is the difference between a @task and a @flow in Prefect?
# ------------------------------------------------------------------
# A @flow is the primary container and orchestrator of your pipeline. It serves as the 
# entry point that establishes the workflow's context, manages execution states, tracks 
# total execution time, and coordinates dependencies. Every Prefect execution must start 
# with a flow.
#
# A @task represents a discrete, isolated unit of work within a flow (like an individual 
# step in an ETL pipeline: Extract, Transform, or Load). Tasks cannot run independently 
# outside of an active flow context. They receive advanced orchestration capabilities 
# such as individual retry policies, separate execution caching, specific tagging, and 
# fine-grained success or failure tracking inside the Prefect UI.
#
# Would you decorate a pure, in-memory temperature conversion helper function with @task? 
# Why or why not?
# --------------------------------------------------------------------------------------
# No, I would not decorate this helper function with @task.
#
# A pure, in-memory calculation with no I/O operations executes instantaneously and carries 
# zero risk of transient network failure. Registering it as a Prefect task introduces 
# unnecessary orchestration overhead—such as state resolution tracking, logging payloads, 
# and UI database writes—which significantly slows down execution performance for millions 
# of records. It is highly efficient to leave it as a regular, native Python function 
# called directly inside an orchestrated task or flow block.

# Prefect Question 2

# The explicit decorator configuration to enforce up to 3 retry attempts 
# with a 30-second delay between each attempt:
@task(name="call_api", retries=3, retry_delay_seconds=30)

# Prefect Question 3

# Where in the UI do you look to understand what went wrong?
# -----------------------------------------------------------
# 1. Open the Prefect UI Dashboard (typically hosted locally at http://localhost:4200).
# 2. Navigate to the "Flow Runs" tab in the left sidebar and click on the specific failed flow run instance.
# 3. Inside the Flow Run dashboard, look at the visual timeline graph or the "Task Runs" section and click 
#    directly on the task labeled 'transform' that is marked with a red 'Failed' status state.
# 4. Navigate to the "Logs" tab inside that specific 'transform' task run view.
#
# What specific information would you expect to find there?
# -----------------------------------------------------------
# I would expect to find the complete, raw Python Exception Traceback (Stack Trace) detailing the exact line 
# of code where the failure occurred, along with the error type (e.g., OpenAI API AuthenticationError, RateLimitError, 
# or a JSONDecodeError). Additionally, I would expect to see the structured orchestration logs generated right 
# before the failure, indicating the exact inputs passed to the 'transform' task or any transient network timeouts.

# --- Production Patterns ---
# Production Question 1

# What does raise_for_status() do?
# ----------------------------------
# The `raise_for_status()` method is a built-in function in the `requests` library that checks the 
# HTTP status code of a response. If the server returns an error code—specifically an HTTP 4xx 
# (Client Error) or 5xx (Server Error)—it automatically raises an explicit `requests.exceptions.HTTPError` 
# exception. If the request was successful (2xx status), it does nothing.
#
# Why is it better than checking status codes and printing an error?
# ----------------------------------------------------------------------
# It is better because it follows the production pattern of "fail-fast." Printing an error with 
# a standard `print()` statement suppresses the failure, masking it as a successful execution 
# to the Python interpreter and the Prefect orchestrator. On the other hand, raising an actual 
# exception interrupts the flow instantly, logs a clear stack trace, and accurately reports the 
# true health and status of the infrastructure.
#
# What happens to downstream tasks in each case when the API returns a 500 error?
# ---------------------------------------------------------------------------------
# Case 1: Using if response.status_code != 200: print("error")
# The function will print "error" to standard output, but then it will return a value (likely 
# `None` or an empty payload) and exit cleanly with a success state. The Prefect orchestrator will 
# mark this task as 'Completed'. Downstream tasks (Transform and Load) will still trigger, receive 
# the empty or invalid payload, and inevitably crash with cryptic errors like `TypeError` or 
# `KeyError`, masking the root cause of the failure.
#
# Case 2: Using raise_for_status()
# The method will immediately throw an `HTTPError` exception. The Prefect orchestrator captures 
# this exception and instantly marks the Extract task as 'Failed'. Because the pipeline broke 
# cleanly at the root, Prefect prevents downstream tasks from executing entirely, marking them 
# as 'UpstreamFailed'. This saves cloud compute resources, avoids token costs, and leaves an 
# unmistakable diagnostic trace in the UI.

# Production Question 2

# What does overwrite=True protect you from in this scenario?
# -------------------------------------------------------------
# It guarantees that the pipeline run is fully idempotent. When you fix the bug and restart 
# the workflow from the beginning, the pipeline re-extracts the data, re-transforms it, and 
# pushes it to the exact same cloud path (`final/{today}/weather_etl.json`). Setting `overwrite=True` 
# forces Azure Blob Storage to cleanly replace the old, incomplete, or corrupted file artifact 
# from the previous failed run with the new, successful dataset in a single atomic transaction.
#
# What would happen without it?
# --------------------------------
# Without `overwrite=True`, Azure Blob Storage's default safety constraints would kick in, causing 
# the `upload_blob()` call to throw a `ResourceExistsError` exception. This means your pipeline would 
# execute perfectly through the Extract and Transform phases, only to crash at the final Load step 
# because a file from the previous failed attempt already occupies that specific date path. To fix 
# it, you would be forced to manually delete the corrupted blob from the Azure Portal before every 
# pipeline re-run, which breaks automation workflows.

# Production Question 3

from prefect import task, get_run_logger

@task(name="log_load_telemetry")
def load_records_stub(records: list, blob_path: str) -> None:
    """
    A task stub demonstrating structured logging inside a Prefect runtime context.
    """
    logger = get_run_logger()
    logger.info(f"Successfully loaded {len(records)} records into cloud storage path: '{blob_path}'.")