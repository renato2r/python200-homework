Cloud Concepts Question 1
What is the core economic model of cloud computing, and how does it differ from owning your own servers?

Answer:
Cloud computing uses a pay-as-you-go model, pretty much like a monthly electric bill. This is completely different from running your own data center because it turns massive upfront hardware investments (CapEx) into predictable, ongoing operational costs (OpEx).

Cloud Concepts Question 2
What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.
Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

Answer:
Vertical scaling is basically upgrading a single machine by adding things like more RAM or a faster GPU. Horizontal scaling is just adding more separate machines to work together and share the load.

Scenario 1 (Web App): Horizontal Scaling, because spinning up more servers is the best way to handle a massive spike in concurrent traffic without crashing the site.

Scenario 2 (Data Scientist Model): Vertical Scaling, since a heavy training script needs a beefier machine with a faster GPU to process everything faster.

Scenario 3 (Data Pipeline): Horizontal Scaling, because you can break the 10,000 files into smaller batches and process them simultaneously across multiple nodes.

Cloud Concepts Question 3
Before writing your definitions, classify each item in the list below as IaaS, PaaS, or SaaS. One sentence of reasoning is enough for each.

Gmail

Azure Virtual Machines

Azure App Service

AWS S3 (Simple Storage Service)

GitHub Codespaces

Snowflake

Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

Answer:

Gmail: SaaS. It is a finished web app where the user does not have to worry about any backend infrastructure.

Azure Virtual Machines: IaaS. It is just a blank virtual server giving you raw hardware control.

Azure App Service: PaaS. A pre-configured platform that handles the server maintenance so you can just drop your code in.

AWS S3: IaaS. It is basic cloud storage infrastructure meant for saving and fetching raw files.

GitHub Codespaces: PaaS. A ready-to-go development container that lets you start coding immediately without setting up a local environment.

Snowflake: PaaS. A fully managed data tool where you only focus on your tables and SQL queries.

Definitions
IaaS (Infrastructure as a Service): You rent the raw hardware infrastructure. You are fully responsible for managing the OS, handling security patches, and installing software (like Azure Virtual Machines).

PaaS (Platform as a Service): The cloud vendor takes care of the infrastructure and the OS. Your only job is managing your application code and data configurations (like Azure App Service).

SaaS (Software as a Service): The provider handles the whole stack. You just log in as an end-user and manage your own account settings and data inputs (like Gmail).

Cloud Concepts Question 4
What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?

Answer:
It is a specialized data ecosystem built right on top of a regular cloud provider like Azure to take the pain out of data engineering.

The biggest upside is that you gain tons of speed and automation without messing with servers, but the trade-off is that you lose low-level control over the actual hardware and end up paying higher premium fees for that convenience.

Cloud Concepts Question 5
The lesson names two situations where the cloud is probably not the right choice. What are they?

Answer:
The two situations are when you have highly predictable, static workloads where buying your own hardware is cheaper in the long run, and when you have strict data privacy or regulatory requirements that legally force you to keep data on your own physical servers.

Azure Basics Question 1
What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?

Answer:
A subscription handles billing, while a resource group is a project directory for specific assets. CTD shares the single subscription, but your resource group is yours alone.

Azure Basics Question 2
Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?

Answer:
Ephemeral means everything gets deleted as soon as the session closes. The course setup mounts an Azure File Share to save your home directory files permanently.

Azure Basics Question 3
What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?

Answer:
The private key is a secret kept on your machine, while the public key is the lock uploaded to the remote server. It is safe because the server can only be unlocked by your matching private key, which never crosses the network.

Azure Basics Question 4
Run the following command in Cloud Shell without the --output table flag:
az account show
Paste the output into your answer. Then describe in one sentence what changes when you add --output table.

Answer:

{
"environmentName": "AzureCloud",
"homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
"id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
"isDefault": true,
"managedByTenants": [],
"name": "CTD Nonprofit Sponsorship",
"state": "Enabled",
"tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
"user": {
"cloudShellID": true,
"name": "live.com#renato2r@gmail.com",
"type": "user"
}
}

Adding --output table converts the raw JSON string into a clean, human-readable table.
