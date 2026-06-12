VIDEO

https://youtu.be/2tGS9I2QI9A

Infrastructure Cost Estimates (East US, Linux)
Scenario A (Lightweight Compute): \* Component: Standard_B1s VM (1 vCPU, 1 GB RAM)

Hourly Rate: $0.0104 / hour

Monthly Configuration: Running 160 hours per month (8 hours a day, 5 days a week)

Total Scenario A Cost: $1.66 per month

Scenario B (Heavy Analytics Workload):

Component 1 (Compute): Standard_NC6s_v3 VM (6 vCPUs, 1 NVIDIA V100 GPU)

Rate: $0.9561 / hour

Monthly Cost (24/7 for 730 hours): $697.96

Component 2 (Database): Azure SQL Database (General Purpose Tier, 4 vCores)

Rate: ~$0.5048 / hour

Monthly Cost (24/7 for 730 hours): $368.55

Component 3 (Storage): Azure Blob Storage (Standard Hot Tier, 1 TB Block Blob)

Rate: ~$0.020 / GB per month + transactional costs

Monthly Cost: $20.48

Total Scenario B Infrastructure Cost: $1,086.99 per month

Cloud Shell Python Script Execution & Verification
When executing the automated project_08.py script inside the Azure Cloud Shell environment, the terminal generated the following exact output:

=== Monthly Cost Estimates ===
Scenario A (lightweight): $1.66
Scenario B (GPU VM only): $697.96
Scenario B VM costs 420.5x more than Scenario A
Comparing these automated outputs against our manual budget projections, the figures match up with absolute precision. The script accurately isolates the raw compute nodes, calculating the lightweight baseline at $1.66 and the multi-core GPU architecture at exactly $697.96, which flawlessly mirrors the line-item estimates pulled from the standalone Azure Pricing Calculator.

Narrative Reflection & Surprising Findings
Looking closely at these specific dollar amounts, the massive pricing gap between the two environments is eye-opening. Seeing a jump from a negligible $1.66 up to a staggering $1,086.99 per month completely changes how you look at architecture planning.

The most surprising takeaway is the disproportionate distribution of the budget. Even with the adjusted rate, a single GPU-enabled compute instance eats up $697.96—roughly 64% of the entire heavy workload estimate. Omitting this instance drops the budget significantly. Meanwhile, hosting an entire Terabyte of active data inside Azure Blob Storage remains incredibly economical at just $20.48.

These concrete numbers prove why developers must be ruthless about automation guardrails, ensuring idle GPU worker nodes are completely deprovisioned when pipelines finish processing, while storage accounts can safely remain active without blowing up the shared organizational budget.
