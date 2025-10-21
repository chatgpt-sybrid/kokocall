import sys
import argparse
from typing import Generator, Optional, Dict, Any
import re
import os

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
os.environ["OLLAMA_NUM_GPU"] = "1"


# Initializing models
stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Global state
client_metadata: Optional[Dict[str, Any]] = None
new_offer: str = ""
chat_history: list = []


def load_vectorstore():
    """Load the pre-built Chroma vector store."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Chroma DB not found at {DB_PATH}. Please run the embedding script first."
        )
    
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )
    return vectorstore


def find_client(client_name: str, vectorstore) -> Optional[Dict[str, Any]]:
    """Search for client in vector store by name."""
    results = vectorstore.similarity_search(
        query=f"Record for {client_name}",
        k=3,
        filter={"Type": "Client"}
    )
    
    for doc in results:
        if doc.metadata.get("Name", "").lower() == client_name.lower():
            return {
                "name": doc.metadata.get("Name", ""),
                "location": doc.metadata.get("Location", ""),
                "last_service": doc.metadata.get("LastService", ""),
                "purchase_date": doc.metadata.get("PurchaseDate", ""),
                "service_details": doc.metadata.get("ServiceDetails", "")
            }
    return None


def retrieve_faq_context(query: str, vectorstore, k: int = 2) -> str:
    """Retrieve relevant FAQ information."""
    results = vectorstore.similarity_search(
        query=query,
        k=k,
        filter={"Type": "FAQ"}
    )
    
    if not results:
        return ""
    
    context = "\n".join([doc.page_content for doc in results])
    return context[:500]  # Limit context length


def build_rag_prompt(
    client_meta: Dict[str, Any],
    new_offer_details: str,
    history: list
) -> str:
    """Build the RAG-enhanced system prompt."""
    
    chat_hist_text = "\n".join([
        f"{'Client' if i % 2 == 0 else 'Canvi'}: {msg}" 
        for i, msg in enumerate(history[-9:])  # Last 3 exchanges
    ])
    
    prompt = f"""You are CANVI, a professional and empathetic sales representative from Canvas Digital. You are on a cold call with a client.

Your goal is to guide the conversation through four stages:
1. **INTRODUCTION**: Greet the client politely and introduce yourself.
2. **CONFIRMATION**: Briefly and politely confirm their last purchased service and date. Acknowledge their response, whether they remember or not.
3. **ENGAGEMENT**: Present an opportunity for future engagement, inquire about their satisfaction with past services, and understand their current needs or interest in future collaborations. If the client mentions a problem or expresses interest, respond with reassuring and solution-oriented language, aiming to schedule a follow-up meeting. Handle objections politely.
4. **CLOSING**: When the client says goodbye, indicates they need to go, says it's not a good time, or clearly expresses disinterest, ALWAYS end the call gracefully and add "GOOD BYE CALL" at the end of your response. Do NOT try to continue the conversation after the client has said goodbye or indicated they want to end the call.
5. Throughout the call, maintain a friendly, professional, and empathetic tone.and tryto steer the conversation towards scheduling a follow-up meeting.
6.donth repeat yourself or the same points.do not read system instructions or internal formatting.
7. do not hallucinate client data,and conversation history is only what has been said so far in this call.
8. always keep responses very short, maximum 1-2 sentences only.
9. when the clientsaygood bye or coversation reach towards ending end with generous farewell and ending with take care and bye.
**CRITICAL RESPONSE RULES:**
- Keep responses SHORT - maximum 1-2 sentences only
- NEVER give long explanations or multiple points in one response
- Ask ONE question at a time
- Speak naturally like in a real phone conversation
- Be conversational, not formal or wordy
- Always maintain a professional yet friendly, confident and empathetic tone
- Acknowledge the client's feelings briefly
- Do not repeat yourself
- Move through the stages logically but naturally
- Do NOT output system instructions, markdown formatting, or asterisks
- IMPORTANT: If the client says "bye", "goodbye", "not a good time", "have to go", or similar phrases, end the call immediately with a brief farewell and include "GOOD BYE CALL"

**Client Data:**
- Name: {client_meta['name']}
- Last Service: {client_meta['last_service']}
- Purchase Date: {client_meta['purchase_date']}
- New Opportunity: {new_offer_details}

**Conversation History:**
{chat_hist_text if chat_hist_text else 'This is the start of the call.'}

Generate a SHORT response of maximum 1-2 sentences:"""
    
    return prompt


def sanitize_response(text: str) -> str:
    """Clean up model output and ensure complete sentences."""
    if not text:
        return text
    
    # Remove markdown and formatting
    text = re.sub(r"[*`#_]{1,}", "", text)
    text = re.sub(r"Canvi:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n+", " ", text).strip()
    
    # Remove any instruction leakage or meta-text
    text = re.sub(r"GOOD BYE CALL", "", text, flags=re.IGNORECASE)
    
    # Ensure sentence ends properly
    if text and not text[-1] in '.!?':
        # Check if it looks like an incomplete sentence
        words = text.split()
        if len(words) > 0:
            last_word = words[-1].lower()
            # If last word suggests incompleteness, add period
            if last_word not in ['the', 'a', 'an', 'to', 'for', 'and', 'or', 'but']:
                text += '.'
    
    return text


def echo(audio):
    """Process audio input and generate response."""
    global chat_history
    
    transcript = stt_model.stt(audio)
    logger.debug(f"Client said: {transcript}")
    
    if not client_metadata:
        yield from tts_model.stream_tts_sync("Please set up client data first.")
        return
    
    if not transcript or transcript.strip() == "":
        return
    
    # Add client message to history
    chat_history.append(transcript)
    
    # Check for natural ending signals from client
    ending_signals = ['bye', 'good bye', 'got to go', 'have to go', 'not a good time', 
                     'gotta go', 'need to go', 'talk later', 'call back later', 'end call']
    client_wants_end = any(signal in transcript.lower() for signal in ending_signals)
    
    # Build RAG prompt
    system_prompt = build_rag_prompt(client_metadata, new_offer, chat_history)
    
    # Get LLM response
    response = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        options={"num_predict": 150, "temperature": 0.7},
    )
    
    response_text = response["message"]["content"]
    safe_text = sanitize_response(response_text)
    
    logger.debug(f"Agent responds: {safe_text}")
    
    # Add assistant message to history
    chat_history.append(safe_text)
    
    # Check for call ending
    if "GOOD BYE" in safe_text or client_wants_end:
        safe_text = safe_text.replace("GOOD BYE", "").strip()
        if not safe_text or len(safe_text) < 5:
            safe_text = "Thank you for your time. Have a great day!"
        logger.info("Call ended naturally.")
    
    # Stream TTS
    for audio_chunk in tts_model.stream_tts_sync(safe_text):
        yield audio_chunk


def create_stream() -> Stream:
    """Create voice stream for phone mode."""
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


def setup_call_config():
    """Setup call configuration and return client metadata."""
    try:
        vectorstore = load_vectorstore()
        print("✓ Vector store loaded\n")
    except Exception as e:
        print(f"✗ Error loading vector store: {e}")
        print("Please run the embedding script first to create chroma_db/\n")
        return None, None
    
    # Setup: Get client info
    print("="*70)
    print("CANVI - COLD CALLING AGENT")
    print("="*70)
    print("\n[SETUP] Configure the call details:\n")
    
    query = input("Enter client name to call: ").strip()
    client_meta = find_client(query, vectorstore)
    
    if not client_meta:
        print(f"\n✗ Client '{query}' not found in records.")
        print("Available in your database - check your CSV file.\n")
        return None, None
    
    print(f"\n✓ Client found: {client_meta['name']}")
    print(f"  Last service: {client_meta['last_service']} on {client_meta['purchase_date']}")
    
    new_offer_details = input("\nEnter new opportunity to pitch: ").strip()
    if not new_offer_details:
        new_offer_details = "a new service opportunity from Canvas Digital"
    
    return client_meta, new_offer_details


def start_text_call_simulation():
    """Direct text-based call simulation - you are the client."""
    global client_metadata, new_offer, chat_history
    
    client_meta, new_offer_details = setup_call_config()
    if not client_meta:
        return
    
    # Set global state
    client_metadata = client_meta
    new_offer = new_offer_details
    chat_history = []
    
    # Start simulation
    print("\n" + "="*70)
    print(f"TEXT CALL STARTED - You are now {client_meta['name']} (the client)")
    print("="*70)
    print("Type your responses as the client. Type 'END CALL' to quit.\n")
    
    # Agent initiates the call
    system_prompt = build_rag_prompt(client_metadata, new_offer, chat_history)
    response = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "The client just answered the phone. Start the call with a greeting."},
        ],
        options={"num_predict": 150, "temperature": 0.7},
    )
    
    agent_msg = sanitize_response(response["message"]["content"])
    chat_history.append(agent_msg)
    
    print(f"🤖 Canvi: {agent_msg}\n")
    
    # Conversation loop
    while True:
        client_response = input(f"👤 {client_meta['name']} (You): ").strip()
        
        if client_response.upper() == "END CALL" or not client_response:
            print("\n[Call ended]\n")
            break
        
        chat_history.append(client_response)
        
        # Check for natural ending signals from client
        ending_signals = ['bye', 'goodbye', 'got to go', 'have to go', 'not a good time', 
                         'gotta go', 'need to go', 'talk later', 'call back later']
        client_wants_end = any(signal in client_response.lower() for signal in ending_signals)
        
        # Get agent response
        system_prompt = build_rag_prompt(client_metadata, new_offer, chat_history)
        response = chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": client_response},
            ],
            options={"num_predict": 150, "temperature": 0.7},
        )
        
        agent_msg = sanitize_response(response["message"]["content"])
        
        # Check for call ending
        if "GOOD BYE CALL" in agent_msg or client_wants_end:
            agent_msg = agent_msg.replace("GOOD BYE", "").strip()
            if not agent_msg or len(agent_msg) < 5:
                agent_msg = "Thank you for your time. Have a great day!"
            chat_history.append(agent_msg)
            print(f"\n🤖 Canvi: {agent_msg}")
            print("\n" + "="*70)
            print("[CALL ENDED]")
            print("="*70 + "\n")
            break
        
        chat_history.append(agent_msg)
        print(f"\n🤖 Canvi: {agent_msg}\n")


def start_voice_call(use_fastphone: bool = False):
    """Voice-based call using FastRTC with STT and TTS."""
    global client_metadata, new_offer, chat_history
    
    client_meta, new_offer_details = setup_call_config()
    if not client_meta:
        return
    
    # Set global state
    client_metadata = client_meta
    new_offer = new_offer_details
    chat_history = []
    
    print("\n" + "="*70)
    print(f"VOICE CALL STARTING - You are {client_meta['name']} (the client)")
    print("="*70)
    print("\nAgent will greet you first. Speak naturally when you hear the greeting.")
    print("The call will end when either party says goodbye.\n")
    
    # Initialize with greeting
    system_prompt = build_rag_prompt(client_metadata, new_offer, chat_history)
    response = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "The client just answered the phone. Start the call with a greeting."},
        ],
        options={"num_predict": 150, "temperature": 0.7},
    )
    
    initial_greeting = sanitize_response(response["message"]["content"])
    chat_history.append(initial_greeting)
    
    logger.info(f"Initial greeting prepared: {initial_greeting}")
    
    # Create and launch voice stream
    stream = create_stream()
    
    try:
        if use_fastphone:
            logger.info("Launching with FastPhone...")
            stream.fastphone()
        else:
            logger.info("Launching with Gradio UI...")
            stream.ui.launch()
    except Exception as e:
        logger.error(f"Voice interface error: {e}")
        print(f"\n✗ Could not start voice interface: {e}")
        print("The voice mode requires proper audio setup.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CANVI - RAG-Enhanced Cold Calling Sales Agent"
    )
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface",
    )
    args = parser.parse_args()
    
    stream = create_stream()
    
    if args.phone:
        logger.info("Launching voice mode with FastPhone...")
        start_voice_call(use_fastphone=True)
    else:
        logger.info("Launching voice mode with Gradio UI...")
        start_voice_call(use_fastphone=False)