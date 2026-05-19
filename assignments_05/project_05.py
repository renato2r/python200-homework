import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# --- ENVIRONMENT SETUP & INITIALIZATION ---
# ==========================================

load_dotenv()

# Verify that the key is present before initializing the client
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ Error: OPENAI_API_KEY not found. Please check your root .env file.")

# Initialize the OpenAI client
client = OpenAI()

# ==========================================
# --- TASK 1: HELPER FUNCTION & PROMPT -----
# ==========================================

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """
    Helper function to request chat completions from the OpenAI API.
    Uses the modern max_completion_tokens parameter to control response length.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400  # Restricts long runaway generations to manage budget
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """
You are an elite, highly encouraging, and strategic Job Application Coach specializing in career transitions. 
Your primary goal is to help professionals translate their past professional experiences, legacy technical skills, 
and background achievements into powerful, modern vocabulary that perfectly aligns with their new target industries.

CRITICAL OPERATIONAL RULES AND GUARDRAILS:
1. SCOPE FOCUS: You must strictly discuss job application materials, including resume bullet points, cover letters, LinkedIn profile optimization, and technical interview preparation. If a user asks questions outside of career development, gently redirect them back to the scope of their application materials.
2. USER DISCRETION REMINDER: You do not possess real-time visibility into local corporate variations or specialized niche industry regulations. Explicitly state that the user must apply their own domain expertise and personal judgment to your suggestions.
3. QUALITY CONTROL: At the end of every comprehensive deliverable (such as a rewritten bullet point or a cover letter block), you must append a brief, standardized reminder instructing the user to thoroughly review, refine, and customize the text to match their authentic voice before submitting it to employers.
"""

# ==========================================
# --- TASK 2: BULLET POINT REWRITER --------
# ==========================================

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """
    Takes a list of raw resume bullet points, sends them to the OpenAI API 
    to be rewritten using action-oriented language, parses the JSON response, 
    and returns a structured list of dictionaries.
    """
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    
    prompt = f"""
You are a professional resume coach helping a career changer.
Rewrite each resume bullet point below to be more specific, results-oriented, and compelling. 
Use strong action verbs (e.g., 'Optimized', 'Streamlined', 'Spearheaded') and imply a sense of scope and scale. 
Do not invent completely fictitious facts or metrics that aren't implied by the original context.

Return your response ONLY as a valid JSON list of objects. Do not include markdown code block wrappers (like ```json).
Each item in the list must contain exactly two keys:
1. "original": The original text provided.
2. "improved": Your updated, high-impact version.

Bullet points to process: {bullet_text}

"""
    messages = [{"role": "user", "content": prompt}]
    raw_response = get_completion(messages, temperature=0.2)
    
    try:
        return json.loads(raw_response.strip())
    except json.JSONDecodeError:
        return [{"original": b, "improved": "Error generating improvement."} for b in bullets]

# ==========================================
# --- TASK 3: COVER LETTER GENERATOR -------
# ==========================================

def generate_cover_letter(job_title: str, background: str) -> str:
    """
    Generates a compelling, high-impact introductory paragraph for a cover letter
    using Few-Shot prompting to capture an active, non-generic tone.
    """
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
The paragraph should be exactly 3-5 sentences: confident, specific, and completely free of clichés (do NOT use phrases like "I am writing to express my enthusiastic interest" or "unique skillset").

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions under pressure using incomplete information — which turns out to be excellent training for data analysis. I recently completed a data analytics program where I built dashboards tracking patient outcomes across departments. I'm excited to bring that combination of clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions get made by people who had never processed a wire transfer or resolved a failed ACH batch. That frustration turned into curiosity, and two years of self-teaching Python later, I'm ready to be on the other side of those decisions. I'm applying to [Company] because your work on payment infrastructure is exactly where my domain expertise and new technical skills intersect.

Now write a highly tailored opening paragraph for this person:
Role: {job_title}
Background: {background}

Opening:
"""
    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)

# ==========================================
# --- TASK 4: MODERATION CHECK -------------
# ==========================================

def is_safe(text: str) -> bool:
    """
    Evaluates untrusted user input using OpenAI's moderation platform.
    """
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    
    if flagged:
        print("\n⚠️ [Moderation Alert] Your input violates our system safety guidelines.")
        print("   Please rephrase your request to focus strictly on professional career content.\n")
        return False
    return True

# ==========================================
# --- TASK 5: THE CHATBOT LOOP -------------
# ==========================================

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' or 'exit' at any time to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break
            
        # 3. Skip empty input
        if not user_input:
            continue
            
        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message
            
        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            
            if raw_bullets:
                print("\nJob Application Helper: Processing your bullet points...")
                improved_list = rewrite_bullets(raw_bullets)
                
                print("\n=== Rewritten Resume Bullets ===")
                for idx, item in enumerate(improved_list, start=1):
                    print(f"\n[Bullet #{idx}]")
                    print(f"  ❌ Original: {item.get('original')}")
                    print(f"  ✅ Improved: {item.get('improved')}")
                
                print("\n💡 Reminder: Please review and edit these bullet points to align with your project metrics before using them.")
                print("-" * 40 + "\n")
            else:
                print("\nJob Application Helper: No bullet points were provided.\n")
                
        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            
            if job_title and background:
                print("\nJob Application Helper: Drafting your opening paragraph...")
                letter_opening = generate_cover_letter(job_title, background)
                
                print("\n=== Cover Letter Opening Paragraph ===")
                print(letter_opening)
                print("\n💡 Reminder: Make sure to review, adapt, and customize this draft to match your authentic voice.")
                print("-" * 40 + "\n")
            else:
                print("\nJob Application Helper: Missing job title or background description. Let's try again.\n")
                
        # 7. Otherwise, handle it as a regular chat turn
        else:
            # Append the user's message to the ongoing conversation history
            messages.append({"role": "user", "content": user_input})
            
            # Request context-aware completion from the coach
            reply = get_completion(messages, temperature=0.7)
            
            # Print the reply to the terminal
            print(f"\nJob Application Helper: {reply}\n")
            
            # Append the assistant's reply back to memory to maintain the conversational state
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()
    
# ==========================================
# --- TASK 6: ETHICS REFLECTION ------------
# ==========================================
# Format Chosen: Option A — Comment block

# Reflection Questions Addressed:
# 1. Potential biases in training data regarding communication styles/backgrounds.
# 2. Risks of a job-seeker submitting unreviewed bot output directly to an employer.
# 3. Additional professional guardrails for deployment.

# Reflection:
# The foundational models powering this assistant are heavily trained on digitized corporate text, 
# which inherently biases their recommendations toward Western, tech-centric, and highly assertive 
# communication styles. This can penalize qualified candidates from cultures that value professional 
# humility, or individuals transitioning from traditional fields where merit is not traditionally 
# communicated through aggressive, metric-driven action verbs. 
#
# If a job-seeker submits the AI's output directly without review, they risk presenting "hallucinated" 
# metrics, inflated technical proficiencies, or an algorithmic tone that lacks authentic human voice, 
# which can severely damage their professional credibility during an interview. 
#
# If I were deploying this tool professionally, I would implement a strict UI-blocking review gate: 
# the interface would explicitly prevent the user from copying the text or clicking "Export" until 
# they have actively interacted with the text box to edit or manually confirm the accuracy of 
# the generated accomplishments.