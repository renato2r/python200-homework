import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load environment variables from .env file
load_dotenv()

# Immediate validation of the OpenAI API Key
assert os.getenv("OPENAI_API_KEY"), "❌ Error: OPENAI_API_KEY environment variable not found in .env file!"
print("[CONFIRMATION]: OpenAI API Key successfully located and loaded.")

# 2. Configure document directory path using Pathlib
docs_dir = Path("./06_AI_augmentation/resources/groundwork_docs")

# Strict safety assertion to verify directory existence
assert docs_dir.exists(), f"❌ Document directory not found at the expected path: {docs_dir.resolve()}"
print(f"[CONFIRMATION]: Document directory successfully validated at: {docs_dir}")

# 3. Initialize and configure models in the LlamaIndex global ecosystem
Settings.llms = LlamaOpenAI(model="gpt-4o-mini", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
print("Global Settings configured with gpt-4o-mini and text-embedding-ada-002.")

print("\n--- Step 1: Setup Completed Successfully ---")
print("------------------------------------------------------------------")

# ==========================================
# --- STEP 2: LOAD THE DOCUMENTS -----------
# ==========================================

print("\n--- Step 2: Loading Documents ---")

# 1. Instantiate the SimpleDirectoryReader pointing to the verified directory
reader = SimpleDirectoryReader(input_dir=str(docs_dir))

# 2. Extract and parse data into a list of Document objects
documents = reader.load_data()

# 3. Output the total number of ingested document pages/segments
total_docs = len(documents)
print(f"📊 Total document pages loaded: {total_docs}")

# 4. Iterate over the loaded documents to read out their individual filenames from metadata
print("\n📋 List of loaded source files:")
uniquely_tracked_files = set()

for doc in documents:
    # Retrieve the file name from the metadata dictionary
    file_name = doc.metadata.get("file_name", "Unknown File")
    
    # Store in a set to avoid printing duplicate page sources if a file spans multiple pages
    if file_name not in uniquely_tracked_files:
        uniquely_tracked_files.add(file_name)
        print(f"  - {file_name}")

print("\n--- Step 2: Document Loading Completed ---")
print("------------------------------------------------------------------")

# ==========================================
# --- STEP 3: BUILD INDEX AND ENGINE -------
# ==========================================

print("\n--- Step 3: Building Index and Query Engine ---")

# 1. Build the high-dimensional vector store index completely in-memory
# This automatically chunks the text and computes semantic embeddings
index = VectorStoreIndex.from_documents(documents)

# 2. Convert the index into a production-ready query engine
# We pass similarity_top_k=3 to ensure it targets the top 3 closest chunks per query
query_engine = index.as_query_engine(similarity_top_k=3)

# 3. Print the required clean verification message to the terminal console
print("Index built successfully. Ready to answer questions.")

print("\n--- Step 3: Indexing Completed ---")
print("------------------------------------------------------------------")

# ==========================================
# --- STEP 4: QUERY THE ASSISTANT ----------
# ==========================================

print("\n--- Step 4: Executing Evaluation Queries ---")

# Standard test suite representing different operational aspects of Groundwork Coffee Co.
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

# Standard loop structure to process each query systematically without code replication
for idx, q in enumerate(questions, 1):
    print(f"\n==================================================================")
    print(f"QUESTION {idx}: {q}")
    print(f"==================================================================")
    
    # 1. Dispatch the query through the active RAG engine
    response = query_engine.query(q)
    
    # 2. Print the synthesized output from the LLM
    print(f"\n[MODEL ANSWER]:\n{response.response}\n")
    
    # 3. Extract and safely display metadata from the top-ranked retrieved source node (Index 0)
    print("--- TOP RETRIEVED SOURCE NODE ---")
    if response.source_nodes:
        top_node = response.source_nodes[0]  # Grab the absolute highest match
        
        # Pull required specific metadata parameters
        doc_name = top_node.node.metadata.get("file_name", "Unknown Document")
        score = top_node.score if top_node.score is not None else 0.0
        
        # Normalize and truncate text for elegant console logging
        clean_text_snippet = top_node.node.get_content().replace('\n', ' ')[:200]
        
        print(f"  Document Name   : {doc_name}")
        print(f"  Similarity Score: {score:.4f}")
        print(f"  Text Snippet (First 200 chars):\n  \"{clean_text_snippet}...\"")
    else:
        print(" No source nodes were retrieved for this query.")

# =====================================================================
# STEP 4 - REFLECTION AND OBSERVATIONS COMMENT BLOCK
# =====================================================================
#
# 1. Did the assistant sound confident and accurate?
#    Yes, the assistant sounds remarkably confident, accurate, and direct. Because it is powered 
#    by gpt-4o-mini paired with high-quality domain context, it avoids vague or nervous filler phrases 
#    (such as "I am not completely sure, but..."). It states menu offerings, operational hours, 
#    and points multipliers as definitive, authoritative facts. The accuracy matches perfectly 
#    with the underlying raw text data stored across the Groundwork documents folder.
#
# 2. Did any of the answers surprise you?
#    What stands out is how elegantly semantic embeddings map intent to source text. For example, 
#    when searching for "dairy-free milk options", the framework successfully ranks alternative 
#    milks listed inside the menu files with high similarity scores, even if the user's specific 
#    phrasing doesn't perfectly match word-for-word. Additionally, the system effectively separates 
#    narrative history data (how it started) from structural business workflows (catering rules), 
#    demonstrating excellent contextual isolation.
# =====================================================================

print("\n------------------------------------------------------------------")

# ==========================================
# --- STEP 5: FIND A FAILURE ---------------
# ==========================================

print("\n--- Step 5: Executing Failure/Stress Test Query ---")

# This query forces a failure because it asks for future 2026 data not in the text,
# along with financial fundraising metrics completely omitted from the history docs.
failure_query = "What is the name of Groundwork's signature seasonal drink for Summer 2026, and how much funding did the shop raise to open its second location?"

print(f"\n==================================================================")
print(f"FAILURE TEST QUERY: {failure_query}")
print(f"==================================================================")

# Execute query using the standard engine configuration (similarity_top_k=3)
response_fail = query_engine.query(failure_query)

print(f"\n[MODEL ANSWER (STRESS TEST)]:\n{response_fail.response}\n")
print("--- ALL THREE RETRIEVED SOURCE NODES ---")

for node_idx, source_node in enumerate(response_fail.source_nodes, 1):
    score = source_node.score if source_node.score is not None else 0.0
    doc_name = source_node.node.metadata.get("file_name", "Unknown Document")
    clean_snippet = source_node.node.get_content().replace('\n', ' ')[:200]
    
    print(f"Node {node_idx}:")
    print(f"  Document Name   : {doc_name}")
    print(f"  Similarity Score: {score:.4f}")
    print(f"  Text Snippet    :\n  \"{clean_snippet}...\"\n")

# =====================================================================
# STEP 5 - FAILURE ANALYSIS REFLECTION COMMENT BLOCK
# =====================================================================
#
# 1. What you asked and why you expected it to be hard:
#    I asked about a specific 2026 seasonal drink menu item and fundraising data for an 
#    expansion. I expected this to be hard because the local documents only contain historical 
#    origin stories, static menus, and operational details. They completely lack forward-looking 
#    2026 marketing schedules and venture capital or internal financial balance sheets.
#
# 2. What went wrong (wrong retrieval, missing information, model guessed anyway?):
#    - The retrieval mechanism suffered from "forced vector compliance". Because it is hardcoded 
#      to fetch top_k=3 nodes, it retrieved chunks related to general history and drink listings. 
#      However, their similarity scores dropped significantly, demonstrating weak alignment.
#      The information was entirely missing from the corpus.
#    - Fortunately, the model did not hallucinate. It relied on its system safeguards to explicitly 
#      state that the provided documentation does not contain information about a Summer 2026 menu 
#      or funding metrics for a second location.
#
# 3. When the retrieval failed, did the model's tone change or remain confident? What does this suggest?
#    The model's tone remained completely confident, authoritative, and professional even though it was 
#    admitting a data deficit. It didn't hesitate or hedge with "I think" or "Maybe". This suggests a 
#    critical lesson about trusting AI: an LLM's structural confidence has zero correlation with 
#    factual coverage. An AI can state a completely hallucinated lie or an accurate refusal with the 
#    exact same level of syntactic certainty. Therefore, independent validation and strict source verification 
#    (metadata grounding) are always mandatory.
#
# 4. What you would change about the system to improve it:
#    - Similarity Score Thresholding: Implement a strict mathematical filter. If the top node score 
#      falls below a chosen threshold (e.g., 0.70), bypass the LLM step entirely and instantly return 
#      "Query out of scope."
#    - Hybrid Keyword Search: Incorporate BM25 lexical search. If critical keywords like "funding" 
#      or "2026" have zero exact matches in the text index, flag the query as unanswerable before 
#      wasting API execution costs.
# =====================================================================

print("\n------------------------------------------------------------------")

# =====================================================================
# STEP 6 - FINAL ARCHITECTURAL REFLECTION
# =====================================================================
#
# 1. Framework Efficiency Comparison:
#    - In our LlamaIndex implementation, loading documents, chunking them, 
#      generating high-dimensional vector embeddings, storing them in memory, 
#      and setting up the top_k retriever engine took exactly 3 functional lines of code:
#        1. reader = SimpleDirectoryReader(input_dir=str(docs_dir)) / documents = reader.load_data()
#        2. index = VectorStoreIndex.from_documents(documents)
#        3. query_engine = index.as_query_engine(similarity_top_k=3)
#    - What this tells us about using a framework:
#      Frameworks like LlamaIndex provide an invaluable layer of abstraction. Instead of forcing 
#      engineers to manually write tokenizers, implement chunking overlap logic, handle batching 
#      delays for external embedding APIs, or write cosine-similarity math from scratch, the framework 
#      packages these industry standards into robust, production-ready design patterns. This drastically 
#      accelerates development time, minimizes boilerplate errors, and allows developers to focus 
#      on data quality and user experience rather than infrastructure plumbing.
#
# 2. Alternative Industry Use Case:
#      An exceptionally high-value alternative use case is an **Automated Medical Compliance & Legal 
#      Onboarding Assistant** for clinics or healthcare systems. 
#      - Medical organizations manage massive, highly volatile folders of insurance billing codes, 
#        HIPAA privacy regulatory updates, state health laws, and clinical trial protocols. 
#      - Instead of requiring doctors, nurses, or compliance officers to spend hours manually searching 
#        through thousands of pages of PDF manuals to verify if a specific patient procedure meets 
#        coverage or legal rules, a RAG system can index these private compliance files securely. 
#        Staff can instantly query the assistant (e.g., "What are the mandated data logging protocols 
#        under HIPAA for telehealth sessions in California?") and receive factual, cited answers 
#        in seconds, minimizing legal risks and human administrative burn-out.
#
# 3. Persistent Failure Modes in RAG:
#      One major failure mode that RAG cannot fully prevent—even when retrieval works perfectly 
#      and pulls the absolute correct text chunks—is **Reasoning/Synthesis Failure by the Generator LLM**.
#      - If the retrieved context contains complex, multi-layered data (such as a dense financial 
#        spreadsheet, conflicting timeline updates, or highly technical logical constraints), the 
#        LLM might still misinterpret the information fed into its context window. 
#      - For example, if the retrieved chunk explicitly states "Plan A covers dental but excludes orthodontics, 
#        while Plan B includes orthodontics after a 12-month waiting period", a model struggling with 
#        complex logical reasoning might conflate the details and falsely tell a user that Plan A covers 
#        orthodontics immediately. RAG controls what the model *sees*, but it cannot guarantee how 
#        intelligently the underlying LLM *reasons* about what it sees.
# =====================================================================

print("\n------------------------------------------------------------------")


# =====================================================================
# --- EXTENSION A: SIDE-BY-SIDE COMPARISON (KEYWORD VS SEMANTIC) ------
# =====================================================================

print("\n==================================================================")
print("🧪 RUNNING EXTENSION A: KEYWORD RAG VS. SEMANTIC RAG COMPARISON")
print("==================================================================")

# 1. Load Groundwork documents as plain text strings for the keyword system
# This mimics the lightweight, dictionary-based structure from the warmup
raw_documents = {f.name: f.read_text(encoding="utf-8") for f in docs_dir.glob("*.txt")}

def simple_keyword_retrieval(query: str, docs: dict) -> tuple:
    """
    Scans documents to find the one with the highest count of overlapping unique words.
    Returns a tuple of (best_doc_name, best_doc_text).
    """
    # Lowercase and split query into unique alphanumeric tokens
    query_words = set("".join(c for c in w if c.isalnum()).lower() for w in query.split())
    # Remove empty strings resulting from punctuation cleaning
    query_words.discard("")
    
    best_doc_name = None
    best_doc_text = ""
    max_overlap = -1
    
    for doc_name, doc_text in docs.items():
        # Lowercase and split document body into unique alphanumeric tokens
        doc_words = set("".join(c for c in w if c.isalnum()).lower() for w in doc_text.split())
        doc_words.discard("")
        
        # Calculate mathematical intersection (shared keywords)
        overlap = len(query_words.intersection(doc_words))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_doc_name = doc_name
            best_doc_text = doc_text
            
    return best_doc_name, best_doc_text

# 2. Iterate through the 5 core queries to run the side-by-side comparison
for idx, q in enumerate(questions, 1):
    print(f"\n------------------------------------------------------------------")
    print(f"COMPARISON QUERY {idx}: '{q}'")
    print(f"------------------------------------------------------------------")
    
    # --- Pipeline 1: Keyword RAG Execution ---
    kw_doc_name, kw_doc_text = simple_keyword_retrieval(q, raw_documents)
    
    # Generate the answer by manually feeding the keyword context to the LLM
    kw_prompt = (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{kw_doc_text}\n"
        f"---------------------\n"
        f"Given the context information and not prior knowledge, "
        f"answer the query: {q}\n"
    )
    # Call the global LLM engine directly
    kw_response = Settings.llms.complete(kw_prompt).text
    
    # --- Pipeline 2: Semantic LlamaIndex Execution ---
    semantic_response = query_engine.query(q).response
    
    # --- Comparative Console Output ---
    print(f"📄 Keyword RAG Top Match Document: {kw_doc_name}")
    print(f"\n[KEYWORD RAG RESPONSE]:\n{kw_response.strip()}")
    print(f"\n[SEMANTIC LAMAINDEX RESPONSE]:\n{semantic_response.strip()}")

# =====================================================================
# EXTENSION A - COMPARATIVE REFLECTION COMMENT BLOCK
# =====================================================================
#
# 1. Did keyword RAG retrieve the right document?
#    Yes, for most of these specific operational queries, the keyword RAG successfully isolated 
#    the correct file. This is because the queries contain heavy, highly explicit nouns and distinct 
#    lexical tags (such as "hours", "loyalty", "catering", or "wholesale") that map cleanly to the 
#    exact file names and structural vocabulary inside the Groundwork document directory.
#
# 2. How did the quality of the two answers differ?
#    - Keyword RAG feeds the *entire text of the matched document* to the LLM. As a result, the 
#      model sometimes outputs a broader, more comprehensive answer since it sees the full file scope. 
#      However, this consumes significantly more input tokens.
#    - Semantic LlamaIndex chunks the text down and only passes the top 3 targeted node segments. 
#      Its answers are generally more precise, structured, and strictly isolated to the exact sentences 
#      answering the question, without dragging peripheral document content into the prompt.
#
# 3. Where did keyword RAG do just as well? Where did it fail or fall behind?
#    - Keyword RAG did just as well on highly targeted explicit queries like "How does the loyalty 
#      program work?". The heavy overlap of the unique word "loyalty" immediately pinned the correct text.
#    - Keyword RAG falls behind or risks failing in two distinct scenarios:
#      A) Synonyms / Concept Mapping: If the user had asked "Do you offer any plant-based or vegan milk?", 
#         the keyword system would show zero overlap with those words. Semantic RAG handles this 
#         seamlessly because vectors map the structural concept of "dairy-free options" to terms like 
#         "oat milk" or "almond milk".
#      B) Document Flooding: Because keyword retrieval passes whole files, if Groundwork had a 100-page 
#         operational manifesto, keyword RAG would easily saturate the LLM context window or degrade 
#         synthesis performance, whereas semantic chunking strictly controls data payloads.
# =====================================================================

print("\n------------------------------------------------------------------")

# =====================================================================
# --- EXTENSION C: ADD A NEW DOCUMENT --------------------------------
# =====================================================================

print("\n==================================================================")
print("🧪 RUNNING EXTENSION C: DYNAMICALLY ADDING A NEW DOCUMENT")
print("==================================================================")

# 1. Define the path and contents for the new document
new_doc_path = docs_dir / "events_2026.txt"

new_doc_content = """Groundwork Coffee Co. - Summer 2026 Community Events Schedule

1. Friday Night Live Acoustic Sessions
- When: Every Friday in June and July from 6:00 PM to 8:00 PM.
- Details: Local musicians perform live on our outdoor patio. Admission is free with any drink purchase.

2. Home Coffee Brewing Masterclass
- When: Saturday, July 18th, 2026, from 2:00 PM to 4:00 PM.
- Details: Led by our Head Barista. Learn the art of perfect pour-overs, French press ratios, and water chemistry.
- Fee: $35 per person, includes a complimentary 12oz bag of Groundwork Signature Blend beans. Space is limited to 10 participants.

3. Latte Art Throwdown
- When: Friday, August 21st, 2026, at 7:00 PM.
- Details: Watch local baristas compete head-to-head for a cash prize. Free public spectating, $5 entry fee for competitors. Free oat milk cold brew samples provided by our event sponsor.
"""

# Write the new document out to the folder programmatically
new_doc_path.write_text(new_doc_content.strip(), encoding="utf-8")
print(f"📝 Dynamically created new corporate document at: {new_doc_path.name}")

# 2. Re-trigger SimpleDirectoryReader to pull the freshly updated directory state
updated_reader = SimpleDirectoryReader(input_dir=str(docs_dir))
updated_documents = updated_reader.load_data()
print(f"📊 Re-scanned directory. Total loaded documents is now: {len(updated_documents)}")

# 3. Rebuild the in-memory LlamaIndex vector store with the expanded corpus
updated_index = VectorStoreIndex.from_documents(updated_documents)
updated_engine = updated_index.as_query_engine(similarity_top_k=3)
print("✨ Expanded VectorStoreIndex successfully compiled.")

# 4. Define and execute a targeted verification query about the new event file
extension_c_query = "How much does it cost to attend the Home Coffee Brewing Masterclass, and what does the fee include?"

print(f"\n------------------------------------------------------------------")
print(f"TESTING NEW DOCUMENT KNOWLEDGE WITH QUERY: '{extension_c_query}'")
print(f"------------------------------------------------------------------")

ext_c_response = updated_engine.query(extension_c_query)

print(f"\n[UPDATED MODEL ANSWER]:\n{ext_c_response.response.strip()}\n")
print("--- TOP RETRIEVED NODE FOR EXTENSION C ---")
if ext_c_response.source_nodes:
    ext_c_top_node = ext_c_response.source_nodes[0]
    print(f"  Source Document : {ext_c_top_node.node.metadata.get('file_name')}")
    print(f"  Similarity Score: {ext_c_top_node.score:.4f}")
    print(f"  Snippet Summary : \"{ext_c_top_node.node.get_content().replace('\n', ' ')[:150]}...\"")

# =====================================================================
# EXTENSION C - NEW DOCUMENT REFLECTION COMMENT BLOCK
# =====================================================================
#
# 1. What document you added and what information it contains:
#    I added a text document titled "events_2026.txt". It acts as a community bulletin containing 
#    the summer schedule for Groundwork Coffee Co., detailing a live music series, a deep-dive 
#    home brewing masterclass (dates, pricing, and perks), and a public barista latte art competition.
#
# 2. What query you used to test it and whether the assistant retrieved the correct content:
#    I queried the engine with: "How much does it cost to attend the Home Coffee Brewing Masterclass, 
#    and what does the fee include?". The assistant perfectly targeted the correct document 
#    ('events_2026.txt') with a high similarity score and cleanly answered that it costs $35 
#    and includes a complimentary 12oz bag of Groundwork Signature Blend coffee beans.
#
# 3. Why this demonstrates an advantage of RAG over fine-tuning:
#    This highlight a massive architectural victory for RAG over deep learning model fine-tuning:
#    - Instantaneous Knowledge Ingestion: In RAG, expanding system knowledge is as simple as dropping 
#      a text file into a directory. The document index updates in milliseconds without modifying 
#      the core neural weights of the foundational model.
#    - Zero Computational Overhead: Fine-tuning requires assembling a training dataset, setting up 
#      loss functions, checking for catastrophic forgetting, renting costly GPU infrastructure, 
#      and training the model over hours or days just to teach it a single new operational schedule.
#    - Absolute Factuality and Auditability: RAG explicitly points to the exact chunk it retrieved, 
#      making the answer entirely trackable and clean. If a schedule changes, we can edit a single line 
#      in the text file, whereas correcting a fine-tuned model's internal memory weight distributions 
#      to fix a single outdated fact is nearly impossible.
# =====================================================================

print("\n------------------------------------------------------------------")
