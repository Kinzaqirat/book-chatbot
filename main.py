import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models
from typing import List, Optional
import json
import uuid
import requests

# Load environment variables
load_dotenv()


# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "kinza-saeed-collection"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not QDRANT_HOST:
    raise ValueError("QDRANT_HOST not found in environment variables")

# --- FastAPI App ---
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
openai_client: AsyncOpenAI | None = None
qdrant_client: QdrantClient | None = None


@app.on_event("startup")
async def startup_event():
    global openai_client, qdrant_client
    openai_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    else:
        qdrant_client = QdrantClient(host=QDRANT_HOST, api_key=QDRANT_API_KEY)
    
    # Ensure collection exists
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' exists with {collection_info.points_count} points")
    except Exception as e:
        print(f"Collection not found. Creating '{COLLECTION_NAME}'...")
        try:
            create_qdrant_collection()
            print(f"Collection '{COLLECTION_NAME}' created successfully")
        except Exception as create_error:
            print(f"Error creating collection: {create_error}")


# --- Pydantic Models ---
class Document(BaseModel):
    id: str | int  # Accept both string and integer
    content: str
    metadata: dict = {}


class BulkDocuments(BaseModel):
    documents: List[Document]


class ChatRequest(BaseModel):
    query: str
    top_k: int = 4
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatWithHistoryRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []
    top_k: int = 4
    stream: bool = False


class DeleteDocumentRequest(BaseModel):
    id: str | int  # Accept both string and integer


# --- Helper Functions ---
async def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using Gemini's model."""
    global openai_client
    response = await openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# data = {"query": "What is AI?", "top_k": 3, "stream": False}
# response = requests.post("http://127.0.0.1:8000/chat", json=data)
# print(response.json())

def create_qdrant_collection():
    """Creates a Qdrant collection if it doesn't exist."""
    global qdrant_client
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    return {"message": f"Collection '{COLLECTION_NAME}' created or recreated."}


async def index_document(doc: Document):
    global qdrant_client
    embedding = await get_embedding(doc.content)
    
    # Convert ID to string first
    doc_id = str(doc.id)
    
    # Convert string ID to UUID if it's not already
    try:
        point_id = uuid.UUID(doc_id)
    except ValueError:
        # If it's not a valid UUID, generate one from the string
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
    
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(point_id),
                payload={"content": doc.content, "original_id": doc_id, **doc.metadata},
                vector=embedding,
            )
        ],
    )
    return {"message": f"Document '{doc_id}' indexed successfully.", "point_id": str(point_id)}


async def index_bulk_documents(documents: List[Document]):
    """Index multiple documents at once with parallel embedding generation."""
    global qdrant_client
    
    # Generate embeddings in parallel
    import asyncio
    
    async def process_document(doc: Document):
        embedding = await get_embedding(doc.content)
        
        # Convert ID to string first
        doc_id = str(doc.id)
        
        # Convert string ID to UUID if it's not already
        try:
            point_id = uuid.UUID(doc_id)
        except ValueError:
            # If it's not a valid UUID, generate one
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
        
        return models.PointStruct(
            id=str(point_id),
            payload={"content": doc.content, "original_id": doc_id, **doc.metadata},
            vector=embedding,
        )
    
    # Process documents in parallel (up to 10 at a time to avoid rate limits)
    points = []
    batch_size = 10
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_points = await asyncio.gather(*[process_document(doc) for doc in batch])
        points.extend(batch_points)
        print(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
    
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    return {"message": f"{len(documents)} documents indexed successfully."}


def delete_document(doc_id: str | int):
    """Delete a document by ID."""
    global qdrant_client
    
    # Convert ID to string first
    doc_id_str = str(doc_id)
    
    # Convert string ID to UUID
    try:
        point_id = uuid.UUID(doc_id_str)
    except ValueError:
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id_str)
    
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(
            points=[str(point_id)],
        ),
    )
    return {"message": f"Document '{doc_id}' deleted successfully."}


def list_documents(limit: int = 100, offset: Optional[str] = None):
    """List all documents in the collection."""
    global qdrant_client
    
    result = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    
    documents = [
        {
            "id": point.payload.get("original_id", str(point.id)),
            "point_id": str(point.id),
            "content": point.payload.get("content", ""),
            "metadata": {k: v for k, v in point.payload.items() if k not in ["content", "original_id"]}
        }
        for point in result[0]
    ]
    
    return {
        "documents": documents,
        "count": len(documents),
        "next_offset": result[1]
    }


def clear_collection():
    """Delete all documents from the collection."""
    global qdrant_client
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    create_qdrant_collection()
    return {"message": f"Collection '{COLLECTION_NAME}' cleared successfully."}


async def rag_query(query_text: str, top_k: int = 4, history: List[ChatMessage] = None) -> str:
    """Main RAG query function with optional chat history."""
    global qdrant_client, openai_client
    
    try:
        # Generate query embedding
        print(f"Generating embedding for query: {query_text}")
        query_embedding = await get_embedding(query_text)
        print(f"Embedding generated, dimension: {len(query_embedding)}")
        
        # Check if collection exists and has data
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection has {collection_info.points_count} documents")
        
        if collection_info.points_count == 0:
            return "The knowledge base is empty. Please upload documents first before asking questions."
        
        # Search in Qdrant using query method
        print(f"Searching in collection '{COLLECTION_NAME}'...")
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        print(f"Found {len(search_result.points)} results")
        
        # Extract context from search results
        context_docs = [
            point.payload.get("content", "") 
            for point in search_result.points 
            if point.payload and point.payload.get("content")
        ]
        
        if not context_docs:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing or upload more documents."
        
        context = "\n\n".join(context_docs)
        print(f"Context built from {len(context_docs)} documents")
        
        # Build messages with history
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately and concisely."}
        ]
        
        # Add chat history if provided
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current query with context
        messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {query_text}"})
        
        # Generate response
        print("Generating LLM response...")
        response = await openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        print(f"Response generated: {answer[:100]}...")
        return answer
        
    except Exception as e:
        print(f"Error in rag_query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"RAG query failed: {str(e)}")


async def rag_query_stream(query_text: str, top_k: int = 4, history: List[ChatMessage] = None):
    """Stream responses from the RAG query."""
    global qdrant_client, openai_client
    
    try:
        # Generate query embedding
        query_embedding = await get_embedding(query_text)
        
        # Check if collection has data
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        
        if collection_info.points_count == 0:
            yield "data: " + json.dumps({"content": "The knowledge base is empty. Please upload documents first."}) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # Search in Qdrant using query method
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        # Extract context from search results
        context_docs = [
            point.payload.get("content", "") 
            for point in search_result.points 
            if point.payload and point.payload.get("content")
        ]
        
        if not context_docs:
            yield "data: " + json.dumps({"content": "I couldn't find any relevant information to answer your question."}) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        
        context = "\n\n".join(context_docs)
        
        # Build messages with history
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately and concisely."}
        ]
        
        # Add chat history if provided
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current query with context
        messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {query_text}"})
        
        # Stream response
        stream = await openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            max_tokens=500,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield "data: " + json.dumps({"content": chunk.choices[0].delta.content}) + "\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"Error in rag_query_stream: {str(e)}")
        import traceback
        traceback.print_exc()
        yield "data: " + json.dumps({"content": f"Error: {str(e)}"}) + "\n\n"
        yield "data: [DONE]\n\n"


def check_qdrant_health():
    """Check if Qdrant is accessible."""
    global qdrant_client
    try:
        collections = qdrant_client.get_collections()
        return {
            "status": "healthy",
            "collections": [col.name for col in collections.collections]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "RAG Chatbot is running!"}


@app.get("/health")
async def health_check():
    """Check the health of the service and Qdrant connection."""
    qdrant_status = check_qdrant_health()
    
    # Test embedding generation
    embedding_test = None
    try:
        test_embedding = await get_embedding("test")
        embedding_test = {"status": "ok", "dimension": len(test_embedding)}
    except Exception as e:
        embedding_test = {"status": "error", "error": str(e)}
    
    return {
        "service": "healthy",
        "qdrant": qdrant_status,
        "embedding": embedding_test
    }


@app.get("/test-search")
async def test_search():
    """Test the search functionality."""
    try:
        # Check collection
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        
        if collection_info.points_count == 0:
            return {
                "status": "empty",
                "message": "No documents in collection. Upload some first.",
                "points_count": 0
            }
        
        # Generate test embedding
        test_embedding = await get_embedding("test query")
        
        # Test search
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=test_embedding,
            limit=3,
            with_payload=True
        )
        
        return {
            "status": "success",
            "points_count": collection_info.points_count,
            "search_results": len(search_result),
            "results": [
                {
                    "score": hit.score,
                    "content_preview": hit.payload.get("content", "")[:100]
                }
                for hit in search_result
            ]
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/create-collection")
async def api_create_collection():
    try:
        return create_qdrant_collection()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/upload-document")
async def api_upload_document(doc: Document):
    try:
        return await index_document(doc)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/upload-documents-bulk")
async def api_upload_documents_bulk(bulk_docs: BulkDocuments):
    """Upload multiple documents at once."""
    try:
        return await index_bulk_documents(bulk_docs.documents)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.delete("/delete-document")
async def api_delete_document(request: DeleteDocumentRequest):
    """Delete a specific document by ID."""
    try:
        return delete_document(request.id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/list-documents")
async def api_list_documents(limit: int = 100, offset: Optional[str] = None):
    """List all documents in the collection."""
    try:
        return list_documents(limit, offset)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/clear-collection")
async def api_clear_collection():
    """Delete all documents from the collection."""
    try:
        return clear_collection()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/chat")
async def api_chat(request: ChatRequest):
    """Simple chat endpoint without history."""
    try:
        print(f"Received chat request: {request.query}")
        
        if request.stream:
            return StreamingResponse(
                rag_query_stream(request.query, request.top_k),
                media_type="text/event-stream"
            )
        else:
            response = await rag_query(request.query, request.top_k)
            return {"response": response}
    except Exception as e:
        print(f"Error in api_chat: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Chat failed: {str(e)}"
        )


@app.post("/chat-with-history")
async def api_chat_with_history(request: ChatWithHistoryRequest):
    """Chat with conversation history support."""
    try:
        if request.stream:
            return StreamingResponse(
                rag_query_stream(request.query, request.top_k, request.history),
                media_type="text/event-stream"
            )
        else:
            response = await rag_query(request.query, request.top_k, request.history)
            return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))