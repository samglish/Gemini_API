from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional

from langchain.llms import Gemini
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

app = FastAPI()

# Initialize your Gemini model
llm = Gemini(model="gemini-pro", api_key="AIzaSyCLNoqwvbQpelCVjez6cSy7ZsP1to5OJY0")

# Create a conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
)

@app.post("/chat")
async def chat(request: Request, message: str, history: Optional[list] = None):
    """
    Endpoint to handle user requests and return responses from the Gemini model.

    Args:
        request: HTTP request object.
        message: User's message.
        history: Optional list of previous messages in the conversation.

    Returns:
        JSONResponse containing the Gemini model's response.
    """

    try:
        # Update the conversation history if provided
        if history:
            conversation.memory.load_memory(history)

        # Generate a response using the conversation chain
        response = conversation.run(message)

        # Store the current conversation history in the response
        history = conversation.memory.load_memory()

        return JSONResponse({"response": response, "history": history})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Define additional endpoints for specific tasks or actions
@app.post("/summarize")
async def summarize(text: str):
    """
    Endpoint to summarize a given text using the Gemini model.

    Args:
        text: The text to summarize.

    Returns:
        JSONResponse containing the summarized text.
    """
    try:
        summary = llm(text, temperature=0.5, max_tokens=100)
        return JSONResponse({"summary": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)