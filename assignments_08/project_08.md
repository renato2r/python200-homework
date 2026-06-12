VIDEO

https://youtu.be/2tGS9I2QI9A

Infrastructure Cost Estimates (East US, Linux)
Scenario A (Lightweight Compute): A Standard_B1s VM (1 vCPU, 1 GB RAM) configured to run 160 hours per month (8 hours a day, 5 days a week) generates a total calculated infrastructure cost of $1.66 per month (based on an exact hourly rate of $0.0104).

Scenario B (Heavy Analytics Workload): This high-end pipeline setup includes a GPU-enabled Standard_NC6s_v3 VM (6 vCPUs, 1 NVIDIA V100 GPU) running 24/7 for 730 hours , an Azure SQL Database on the General Purpose tier with 4 vCores , and 1 TB of Standard Hot Blob Storage .

Cloud Shell Python Script Execution & Verification
When executing the automated project_08.py script inside the Azure Cloud Shell environment, the terminal generated the following exact output:

Plaintext
=== Monthly Cost Estimates ===
Scenario A (lightweight): $1.66
Scenario B (GPU VM only): $697.96
Scenario B VM costs 420.5x more than Scenario A
Comparing these automated outputs against our manual budget projections, the figures match up with absolute precision. The script accurately isolates the raw compute nodes, calculating the lightweight baseline at $1.66 and the multi-core GPU architecture at exactly $697.96, which flawlessly mirrors the line-item estimates pulled from the standalone Azure Pricing Calculator.

Discussion & Surprising Findings
The price gap between a basic development sandbox and a production-grade heavy analytics pipeline is massive—jumping from the price of a cup of coffee to over $8000 a year.

The most surprising discovery during this exploration was how expensive GPU compute power actually is.
