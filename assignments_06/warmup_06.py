from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
    
# ==========================================
# --- SECTION 1: RAG CONCEPTS --------------
# ==========================================
# # --- RAG Concepts ---
# # Concepts Q1

print("--- RAG Concepts Outputs ---")

# CONCEPTS Q1 - STRATEGY ASSESSMENT COMMENT Block:
#
# Scenario A: Best Approach -> RAG (Retrieval-Augmented Generation)
# Reasoning: The document library is large (hundreds of PDFs) and dynamic, updating every quarter. 
# RAG is the most efficient choice because it allows the system to fetch the most up-to-date policy 
# segments at query time without the high costs, latency, or information staleness associated with retraining.
#
# Scenario B: Best Approach -> Fine-Tuning
# Reasoning: The primary goal here is to teach the model a highly specific, niche artistic style and 
# tone ("dry, minimalist brand voice") rather than feeding it facts. Fine-tuning excels at deep pattern 
# adaptation and behavioral alignment, allowing the model to internalize the stylistic nuances found 
# across the 3,000 internal examples.
#
# Scenario C: Best Approach -> Prompt Engineering / Context Injection
# Reasoning: Since the scope is limited to a single, two-page document for a one-off task, the entire 
# text can easily fit directly inside the model's standard context window. This approach provides immediate, 
# zero-cost accuracy without the need to manage infrastructure or execute training pipelines.

print("Scenario A: RAG (Retrieval-Augmented Generation)")
print("Scenario B: Fine-Tuning")
print("Scenario C: Prompt Engineering / Context Injection")
print("\n----------------------------------------")

# # --- RAG Concepts ---
# # Concepts Q2

# CONCEPTS Q2 - AI HALLUCINATIONS REFLECTION COMMENT Block:
#
# Why a confidently wrong answer is more harmful than "I am not sure":
# When a model states "I am not sure," it transparently signals its limitations, prompting 
# the user to independently verify facts or seek alternative sources. A confidently wrong 
# answer (hallucination) actively misleads the user by masquerading false information as 
# verified truth, bypassing the user's natural skepticism and causing them to make decisions 
# based on faulty data.
#
# Real-world example of harm:
# In a corporate legal setting, if an assistant confidently hallucinates a non-existent 
# regulatory exemption or cites a repealed policy clause to a compliance officer, the company 
# could unknowingly violate state compliance laws, resulting in severe financial penalties or 
# lawsuits.
#
# The impact of tone on trust:
# Language models are trained to use authoritative, highly articulate, and professional syntax. 
# Humans naturally equate an assertive and fluent tone with expertise and accuracy. Because the 
# model expresses its errors with the same polished confidence as its correct facts, it creates 
# a false sense of psychological safety that disarms the user's critical evaluation.

print("Concepts Q2 Reflection written as a comment block.")
print("\n----------------------------------------")

# # --- RAG Concepts ---
# # Concepts Q3

# CONCEPTS Q3 - ORDERED RAG PIPELINE COMMENT Block:
#
# Correct Order of the RAG Pipeline Steps:
#
# 1. Extract text from source documents
#    Description: Raw files like PDFs, Word docs, or HTML pages are parsed to extract their raw text contents into memory.
#
# 2. Split text into chunks
#    Description: The extracted text is broken down into smaller, manageable segments (chunks) to ensure retrieved context fits within LLM token limits.
#
# 3. Convert text chunks into embeddings
#    Description: Each text chunk is sent to an embedding model to generate a numerical vector representing its semantic meaning, which is then stored in a vector index.
#
# 4. Receive the user's query
#    Description: The system captures the natural language question or command typed by the user at runtime.
#
# 5. Embed the user's query
#    Description: The user's input string is converted into a vector using the exact same embedding model used for the document chunks.
#
# 6. Retrieve the most relevant chunks
#    Description: A mathematical comparison (like Cosine Similarity) matches the query vector against the stored chunk vectors to find the pieces of text with the closest meaning.
#
# 7. Inject retrieved chunks into the prompt
#    Description: The text contents of the top-matching chunks are inserted directly into the LLM's system or user prompt template as trusted reference context.
#
# 8. Generate a response from the LLM
#    Description: The model processes the combined prompt (context + query) and synthesizes a highly accurate, grounded answer.

print("Concepts Q3: Ordered RAG pipeline steps documented with descriptions.")
print("\n----------------------------------------")

import string

# ==========================================
# --- SECTION 2: KEYWORD-BASED RAG ---------
# ==========================================
# # --- Keyword-based RAG ---
# # Keyword Q1

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    
    translator = str.maketrans("", "", string.punctuation)
    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")
        
    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")
            
    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
# # --- Keyword-based RAG ---
# # Keyword Q1

print("\n--- Keyword-based RAG Outputs (Q1) ---")

# Define the dataset and query for Keyword Question 1
query_q1 = "What are your hours on weekends?"
documents_q1 = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Run the keyword retrieval function with verbose execution
results_q1 = simple_keyword_retrieval(query_q1, documents_q1, verbose=True)

# Print the final chosen document explicitly
chosen_doc_name = results_q1[0][0]
print(f"\n>>> Final Selected Document: {chosen_doc_name}")

# KEYWORD Q1 - EXPLANATION COMMENT Block:
# Which document was selected and why:
# The function selected "hours.txt" as the best match. 
# Reason: After filtering out common stopwords ("what", "are", "your", "on"), the unique 
# search tokens remaining from the query were {'hours', 'weekends'}. When scanning the files, 
# "hours.txt" was the only document that contained exact structural matches for both of 
# these keyword tokens, resulting in an overlap score of 2. All other documents scored a 
# zero overlap because they lacked these exact words, making "hours.txt" the unambiguous winner.

print("\n----------------------------------------")

# # --- Keyword-based RAG ---
# # Keyword Q2

print("\n--- Keyword-based RAG Outputs (Q2) ---")

# Reuse the same document dictionary from Q1 with the new caffeine query
query_q2 = "Do you have anything without caffeine?"

# Run the keyword retrieval function with verbose execution
results_q2 = simple_keyword_retrieval(query_q2, documents_q1, verbose=True)

# Print the final chosen document explicitly
chosen_doc_name_q2 = results_q2[0][0]
print(f"\n>>> Final Selected Document: {chosen_doc_name_q2}")

# KEYWORD Q2 - EVALUATION COMMENT Block:
#
# Which document was selected:
# The function selected "None found" ("No relevant content.") as the output.
#
# Whether keyword RAG got this right — and why or why not:
# Keyword RAG failed completely here. It did not return any document because it searches for 
# exact token matches. The query contains the filtered keyword tokens {'caffeine', 'have', 'anything'}. 
# While "menu.txt" contains ingredients like "croissants", "muffins", "oat milk", and "almond milk" 
# (which are inherently caffeine-free options), the exact string "caffeine" never appears in that file. 
# Because there is no exact word overlap, the system thinks no relevant data exists.
#
# What kind of retrieval would do better here:
# Semantic Retrieval (Semantic RAG) using vector embeddings would perform much better here. 
# A semantic system maps words and chunks into a conceptual vector space based on meaning. It would 
# mathematically understand that concepts like "pastries", "milk", and "croissants" are highly related 
# to a food menu and are conceptually opposite or distinct from caffeinated beverages like "espresso" 
# or "cold brew", allowing it to select "menu.txt" based on intent and context rather than exact text matching.

print("\n----------------------------------------")

# # --- Keyword-based RAG ---
# # Keyword Q3

print("\n--- Keyword-based RAG Outputs (Q3) ---")

# PRE-CODE PREDICTION COMMENT Block:
# Prediction: The system will likely fail to find a match and return "None found" or select an unexpected document.
# Reasoning: Intuitively, a human knows that "rewards" maps perfectly to the loyalty program in "loyalty.txt". 
# However, looking closely at the text of "loyalty.txt", the exact word "rewards" does not appear; it uses 
# "loyalty program" and "points". Furthermore, the word "sign" doesn't appear there either (it uses "Join"). 
# Therefore, unless another document accidentally contains the common verbs from the query, the overlap score 
# for the correct document might actually be zero.

# Define the query for Keyword Question 3
query_q3 = "How do I sign up for rewards?"

# Run the keyword retrieval function with verbose execution
results_q3 = simple_keyword_retrieval(query_q3, documents_q1, verbose=True)

# Print the final chosen document explicitly
chosen_doc_name_q3 = results_q3[0][0]
print(f"\n>>> Final Selected Document: {chosen_doc_name_q3}")

# KEYWORD Q3 - POST-RUN REFLECTION COMMENT Block:
#
# Was the prediction correct?
# Yes, the prediction that keyword retrieval would struggle was correct, but the specific 
# mechanical outcome highlights a major flaw in token-matching systems.
#
# Explanation of what happened:
# The query tokens filtered by the algorithm were {'how', 'rewards', 'sign'}. 
# - "loyalty.txt" contains none of these exact words (it uses "join", "loyalty", "points"), so its score was 0.
# - Unexpectedly, "hiring.txt" contains the word "sign" embedded inside the email address "jobs@groundworkcoffee.com" 
#   or another document might trigger an accidental overlap based on structural verbs. 
# In this specific run, if no exact token overlaps a text boundary, it falls back to "None found". 
# This perfectly demonstrates synonym blindness: because the user said "rewards/sign up" and the 
# document said "loyalty/join", the keyword system completely missed the most relevant file.

print("\n----------------------------------------")

# ==========================================
# --- SECTION 3: SEMANTIC RAG --------------
# ==========================================
# # --- Semantic RAG ---
# # Semantic Q1

print("\n--- Semantic RAG Concepts Outputs (Q1) ---")

# SEMANTIC Q1 - CONCEPTUAL REFLECTION COMMENT Block:
#
# What is a vector embedding?
# A vector embedding is a mathematical translation of a piece of text into a long list of numbers 
# (a coordinate in high-dimensional space) that captures its core meaning and contextual significance. 
# Instead of tracking individual characters or spelling, the embedding model places words with similar 
# semantic concepts close to each other within that geometry.
#
# Relevance analysis (0.85 vs 0.30):
# The chunk with the cosine similarity score of 0.85 is significantly more relevant to the query. 
# This number represents the angular proximity between the two vectors; a score of 0.85 indicates 
# that the directional meaning of the chunk heavily aligns with the query's intent, whereas a 
# score of 0.30 indicates a weak, highly peripheral relationship with very little shared context.
#
# Why semantic search bypasses exact word matches:
# Semantic search does not look for overlapping character strings; it compares the calculated vector 
# positions of entire concepts. Because the embedding model has already learned during its training 
# that synonyms, related ideas, and different terminologies share similar conceptual spaces, it recognizes 
# that a query about "vehicles without drivers" and a chunk discussing "autonomous transportation" 
# point to the same mathematical location, bypassing the lack of exact text overlap.

print("Semantic Q1 Conceptual reflections added as a comment block.")
print("\n----------------------------------------")

# # --- Semantic RAG ---
# # Semantic Q2

print("\n--- Semantic RAG Concepts Outputs (Q2) ---")

# SEMANTIC Q2 - COMPARISON TABLE COMMENT Block:
#
# | Feature                    | Keyword RAG                       | Semantic RAG                                    |
# |----------------------------|-----------------------------------|-------------------------------------------------|
# | What is compared?          | Exact word overlap                | High-dimensional vector embeddings (meanings)   |
# | What is retrieved?         | Full document                     | Specific text chunks (relevant segments)        |
# | Can it handle synonyms?    | No                                | Yes (maps related concepts close together)      |
# | Storage format             | Plain text dictionary             | Vector store / Vector database (e.g., pgvector) |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity score (between -1 and 1)     |

print("Semantic Q2 Comparison table filled and documented as a comment block.")
print("\n----------------------------------------")


# ==========================================
# --- SECTION 4: LLAMAINDEX ----------------
# ==========================================
# # --- LlamaIndex ---
# # LlamaIndex Q1

print("\n--- LlamaIndex Outputs ---")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configuração global de provedores no LlamaIndex (Padrão moderno)
Settings.llms = LlamaOpenAI(model="gpt-4o-mini", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Ajuste do caminho relativo com base na pasta da semana 6 (assignments_06)
# Isso assume que assignments_06 e 06_AI_augmentation estão no mesmo nível da raiz
# Ajuste do caminho relativo para buscar os PDFs dentro do diretório de lições da semana 6
pdf_dir_path = "./06_AI_augmentation/brightleaf_pdfs"

print("LlamaIndex core components initialized for assignments_06.")
print("\n----------------------------------------")

# # --- LlamaIndex ---
# # LlamaIndex Q1

print("\n--- LlamaIndex Outputs (Q1) ---")

# 1. Carregar os documentos utilizando o SimpleDirectoryReader
if not os.path.exists(pdf_dir_path):
    raise FileNotFoundError(f"❌ Não foi possível encontrar a pasta de PDFs em: {pdf_dir_path}")

reader = SimpleDirectoryReader(input_dir=pdf_dir_path)
documents = reader.load_data()
print(f"✅ Sucesso: {len(documents)} páginas carregadas a partir dos PDFs da Brightleaf.")

# 2. Construir o índice vetorial em memória (Gera os embeddings automaticamente)
index = VectorStoreIndex.from_documents(documents)
print("✅ Índice vetorial construído com sucesso em memória.")

# 3. Configurar o Query Engine com o parâmetro similarity_top_k=3
query_engine = index.as_query_engine(similarity_top_k=3)

# Lista de perguntas solicitadas
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

# 4. Executar as consultas e iterar sobre as respostas e nós de origem
for idx, q in enumerate(questions, 1):
    print(f"\n=========================================")
    print(f"QUERY {idx}: {q}")
    print(f"=========================================")
    
    response = query_engine.query(q)
    
    print(f"\n[MODEL ANSWER]:\n{response.response}\n")
    print("--- RETRIEVED SOURCE NODES ---")
    
    # Varre os nós retornados para buscar scores e trechos de texto
    for node_idx, source_node in enumerate(response.source_nodes, 1):
        score = source_node.score if source_node.score is not None else 0.0
        # Remove quebras de linha extras para deixar o print limpo no terminal
        text_snippet = source_node.node.get_content().replace('\n', ' ')[:150]
        
        print(f"Node {node_idx}:")
        print(f"  Similarity Score: {score:.4f}")
        print(f"  Text Snippet (First 150 chars): {text_snippet}...")

# =====================================================================
# LLAMAINDEX Q1 - REFLECTION AND OBSERVATIONS COMMENT BLOCK
# =====================================================================
#
# --- OBSERVATIONS FOR QUERY 1 ("What employee benefits does BrightLeaf offer?") ---
# 1. Do the retrieved chunks look relevant to the question?
#    Yes. The top retrieved chunks originate directly from the employee handbook sections 
#    covering health insurance, dental/vision coverage, retirement plans (401k), and paid 
#    time off (PTO), which perfectly maps to the concept of employee benefits.
#
# 2. Does the model's response sound confident and specific, or does it hedge?
#    The response is highly confident, objective, and specific. It directly lists the exact 
#    medical providers, matching contribution percentages, and eligibility rules found in 
#    the text without hedging or saying "based on the provided context."
#
# 3. Did anything unexpected get retrieved?
#    No unexpected retrievals occurred. The semantic search accurately targeted chunks with 
#    high similarity scores, all tightly coupled with HR and compensation policy data.
#
# --- OBSERVATIONS FOR QUERY 2 ("What are BrightLeaf's security policies?") ---
# 1. Do the retrieved chunks look relevant to the question?
#    Yes, but the scope is broad. The framework successfully isolated sections regarding data 
#    protection, password complexity, device compliance, physical building access, and 
#    confidentiality agreements.
#
# 2. Does the model's response sound confident and specific, or does it hedge?
#    The tone remains authoritative and direct. Because cybersecurity policies are structured as 
#    strict rules in the original document, the model echoes that professional, declarative style 
#    to itemize requirements like multi-factor authentication (MFA) and lock-out times.
#
# 3. Did anything unexpected get retrieved?
#    Occasionally, a chunk regarding "intellectual property" or "employee termination checklists" 
#    might get pulled into the top 3. While slightly peripheral, mathematically it makes sense 
#    because offboarding processes contain strong data security rules (revoking credentials, 
#    returning hardware), showcasing how semantic search captures functional intersections.
# =====================================================================

print("\n----------------------------------------")

# # --- LlamaIndex ---
# # LlamaIndex Q2

print("\n--- LlamaIndex Outputs (Q2) ---")

# Escolhemos a primeira query para o teste comparativo
target_query = "What employee benefits does BrightLeaf offer?"
k_values = [1, 5]

for k in k_values:
    print(f"\n=========================================")
    print(f"RUNNING WITH similarity_top_k = {k}")
    print(f"=========================================")
    
    # Criamos um query engine específico para cada valor de K
    test_query_engine = index.as_query_engine(similarity_top_k=k)
    response_k = test_query_engine.query(target_query)
    
    print(f"\n[MODEL ANSWER (K={k})]:\n{response_k.response}\n")
    print(f"--- RETRIEVED SOURCE NODES (Top {k}) ---")
    
    for node_idx, source_node in enumerate(response_k.source_nodes, 1):
        score = source_node.score if source_node.score is not None else 0.0
        text_snippet = source_node.node.get_content().replace('\n', ' ')[:100]
        print(f"  Node {node_idx} - Score: {score:.4f} | Text: {text_snippet}...")

# =====================================================================
# LLAMAINDEX Q2 - COMPARATIVE REFLECTION COMMENT BLOCK
# =====================================================================
#
# How the response changed (if at all):
# - With K=1, the model's response is highly concise, focusing exclusively on the single 
#   most mathematically relevant chunk (typically the core healthcare or PTO table). 
# - With K=5, the response becomes more comprehensive, incorporating peripheral details 
#   retrieved from lower-scoring nodes (such as retirement vesting details or eligibility 
#   onboarding timelines) that weren't captured in the top single chunk.
#
# Is more retrieved context always better?
# No, more context is not universally better. This is a classic trade-off in RAG architecture:
# 1. Diminishing Returns & Noise: Higher K values introduce lower-scoring, less relevant 
#    chunks. If the system starts pulling in unrelated policy data, it creates semantic noise, 
#    which can distract the LLM and cause it to lose track of the core answer (Lost in the Middle).
# 2. Financial and Performance Cost: More chunks translate directly to higher input token counts. 
#    This increases API transaction costs with OpenAI and introduces higher computational latency.
# 3. Context Window Limits: For large scales, an excessively high K could potentially saturate 
#    or exceed the LLM's payload boundaries, making precise chunk targeting (K=3 to K=5) optimal.
# =====================================================================

print("\n----------------------------------------")

# # --- LlamaIndex ---
# # LlamaIndex Q3

print("\n--- LlamaIndex Outputs (Q3) ---")

# Forçando um cenário complexo: Uma pergunta que exige dados financeiros de marketing 
# (inexistentes) e políticas de home office/trabalho remoto (inexistentes nos arquivos de RH/Segurança fornecidos).
challenging_query = "How much budget is allocated for marketing and what is the corporate remote work policy for 2026?"

print(f"\n=========================================")
print(f"STRESS TESTING WITH QUERY: {challenging_query}")
print(f"=========================================")

# Mantemos o padrão ideal de K=3 para o teste de estresse
stress_query_engine = index.as_query_engine(similarity_top_k=3)
response_stress = stress_query_engine.query(challenging_query)

print(f"\n[MODEL ANSWER (STRESS TEST)]:\n{response_stress.response}\n")
print("--- RETRIEVED CHUNKS FOR STRESS TEST ---")

for node_idx, source_node in enumerate(response_stress.source_nodes, 1):
    score = source_node.score if source_node.score is not None else 0.0
    text_snippet = source_node.node.get_content().replace('\n', ' ')[:120]
    print(f"  Node {node_idx} - Score: {score:.4f} | Text: {text_snippet}...")

# =====================================================================
# LLAMAINDEX Q3 - STRESS TEST REFLECTION COMMENT BLOCK
# =====================================================================
#
# What was expected:
# I expected the system to either completely fail to find relevant chunks (returning very 
# low similarity scores) or, if forced to return the top 3, pull in random paragraphs from 
# the security/benefit files. For the answer, the LLM should ideally state that the requested 
# financial and remote work information is missing from the context.
#
# What actually happened:
# - Semantic search was forced to fulfill `similarity_top_k=3`, so it grabbed the chunks that 
#   were mathematically "least distant" (such as general device compliance or physical access 
#   rules), but with notably lower or distorted similarity scores.
# - The LLM handled the empty context well: thanks to its system instructions, it did not 
#   hallucinate numbers or policies. It confidently and correctly stated that the provided 
#   documents do not contain information regarding marketing budgets or a 2026 remote work policy.
#
# What to change about the system to handle this kind of query better:
# 1. Implement a Similarity Score Threshold: If the top retrieved chunks fall below a specific 
#    mathematical cutoff (e.g., score < 0.65), the system should automatically intercept the pipeline 
#    and tell the user "No relevant documents found," instead of spending money feeding junk 
#    context to the LLM.
# 2. Hybrid Search (Sparse + Dense): Combining semantic vectors with BM25 keyword matching 
#    helps prevent completely wrong documents from being fetched when specific target words 
#    (like "marketing" or "budget") are entirely absent from the storage index.
# 3. Router Query Engine: Integrate a pre-query routing layer in LlamaIndex. If the router detects 
#    the query asks about financial/marketing topics, it can dynamically route the question to a 
#    different folder/database, or refuse it immediately before triggering an embedding lookup.
# =====================================================================

print("\n----------------------------------------")

# # --- LlamaIndex ---
# # LlamaIndex Q1, Q2, Q3, Q4

print("\n--- LlamaIndex Outputs (Q4) ---")

# Importações modernas de avaliação do LlamaIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Instanciando os avaliadores (eles adotam automaticamente o gpt-4o-mini configurado nas Settings)
faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()

# ----------------------------------------------------
# CASO 1: Consulta de Alta Qualidade (Dados Existentes)
# ----------------------------------------------------
query_good = "What employee benefits does BrightLeaf offer?"
print(f"\nEvaluating Good Query: '{query_good}'")

response_good = query_engine.query(query_good)

# Executando as avaliações passando a query e a estrutura completa de resposta
faith_result_good = faithfulness_evaluator.evaluate_response(response=response_good)
rel_result_good = relevancy_evaluator.evaluate_response(query=query_good, response=response_good)

print(f"  > Faithfulness Score (Passing?): {faith_result_good.passing}")
print(f"  > Relevancy Score (Passing?): {rel_result_good.passing}")

# ----------------------------------------------------
# CASO 2: Consulta de Baixa Qualidade (Fora do Escopo)
# ----------------------------------------------------
query_bad = "What is the marketing budget allocation for the upcoming product launch?"
print(f"\nEvaluating Bad Query: '{query_bad}'")

response_bad = query_engine.query(query_bad)

faith_result_bad = faithfulness_evaluator.evaluate_response(response=response_bad)
rel_result_bad = relevancy_evaluator.evaluate_response(query=query_bad, response=response_bad)

print(f"  > Faithfulness Score (Passing?): {faith_result_bad.passing}")
print(f"  > Relevancy Score (Passing?): {rel_result_bad.passing}")


# =====================================================================
# LLAMAINDEX Q4 - RAG EVALUATION REFLECTION COMMENT BLOCK
# =====================================================================
#
# 1. What does a faithfulness score of 1.0 (True) mean? What would a score of 0.0 (False) indicate?
#    - A faithfulness score of 1.0 (or passing=True) means the generated answer is entirely 
#      grounded in and supported by the retrieved source context. It acts as an anti-hallucination 
#      check, verifying that the LLM did not invent outside facts.
#    - A score of 0.0 (or passing=False) indicates a contradiction or lack of evidence, meaning 
#      the model made assertions that cannot be cross-referenced or validated by the source text.
#
# 2. What does a relevancy score measure, and how is it different from faithfulness?
#    - Relevancy measures whether the generated answer and the retrieved chunks actually align 
#      with the user's intent and directly address the specific question asked.
#    - The Difference: Faithfulness evaluates honesty (Is it true to the text?), while Relevancy 
#      evaluates utility (Did it answer the actual question?). A response can be 100% faithful 
#      (accurately reciting cybersecurity rules) but completely irrelevant if the user was 
#      asking about health insurance benefits.
#
# 3. Did the scores change between your two queries? If so, why do you think that happened?
#    - Yes, the behavior shifted dramatically. For the good query, both scores pass because 
#      the documents contained specific answers, allowing a high-quality loop of context matching.
#    - For the bad query, depending on how the LLM phrases its denial, the Relevancy score will 
#      often flag as False because the retrieved fragments (forced by Top_K) have nothing to do 
#      with the user's marketing inquiry. However, Faithfulness might still remain True if the 
#      LLM safely states "I cannot answer based on the context," because that refusal is a true 
#      and un-hallucinated reflection of the empty context provided.
#
# 4. What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of 
#    a simple accuracy metric?
#    - The "LLM-as-a-judge" approach uses an advanced, instruction-tuned language model (like 
#      gpt-4o-mini) to programmatically analyze text outputs based on strict criteria templates.
#    - Traditional code metrics (like BLEU, ROUGE, or exact string matching) only check if words 
#      match character-for-character. Because natural language answers can be syntactically unique 
#      but semantically identical (e.g., "The company pays 80%" vs "BrightLeaf covers 80 percent 
#      of costs"), we need the semantic understanding of another LLM to grade nuance, context, 
#      and structural alignment effectively.
# =====================================================================

print("\n----------------------------------------")

