"""
RAG (Retrieval Augmented Generation) system using Qdrant and LangChain.
Enables natural language querying of real estate data.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantRAGSystem:
    """RAG system using Qdrant vector database and LangChain."""
    
    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_client = self._initialize_qdrant()
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
        
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = 1536  # OpenAI embedding dimension
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _initialize_qdrant(self) -> QdrantClient:
        """Initialize Qdrant client connection."""
        try:
            if settings.qdrant_api_key:
                # Cloud/remote Qdrant
                client = QdrantClient(
                    url=f"{'https' if settings.qdrant_use_https else 'http'}://{settings.qdrant_host}:{settings.qdrant_port}",
                    api_key=settings.qdrant_api_key,
                    timeout=30
                )
            else:
                # Local Qdrant
                client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    timeout=30
                )
            
            logger.info("Successfully connected to Qdrant")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            logger.warning("Running without vector database - similarity search will not be available")
            return None
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        if not self.qdrant_client:
            return
        
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Add a document to the vector database."""
        if not self.qdrant_client:
            logger.warning("Qdrant client not available, skipping document addition")
            return
        
        try:
            # Generate embedding
            embedding = self._generate_embedding(text)
            if not embedding:
                return
            
            # Create point
            point = PointStruct(
                id=hash(doc_id),  # Convert string ID to integer
                vector=embedding,
                payload={
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Upsert to collection
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Added document: {doc_id}")
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
    
    def add_submarket_data(self, submarket: Any):
        """Add submarket analysis to vector database."""
        doc_id = f"submarket_{submarket.city}_{submarket.county}"
        
        text = f"""
        Submarket Analysis: {submarket.city}, {submarket.county}, {submarket.state}
        Composite Score: {submarket.composite_score:.3f}
        School Score: {submarket.school_score:.3f} - {len(submarket.schools)} schools analyzed
        Crime Score: {submarket.crime_score:.3f} - Lower crime is better
        Growth Score: {submarket.growth_score:.3f}
        Price Score: {submarket.price_score:.3f}
        """
        
        if submarket.pricing_data:
            text += f"\nMedian Price per SqFt: ${submarket.pricing_data.median_price_per_sqft:.2f}"
        
        if submarket.growth_metrics:
            text += f"\nPopulation Growth (1yr): {submarket.growth_metrics.population_growth_1yr}%"
        
        metadata = {
            "type": "submarket",
            "city": submarket.city,
            "county": submarket.county,
            "state": submarket.state,
            "composite_score": submarket.composite_score
        }
        
        self.add_document(doc_id, text, metadata)
    
    def add_land_listing(self, listing: Any):
        """Add land listing to vector database."""
        doc_id = f"land_{listing.listing_id}"
        
        text = f"""
        Land Listing: {listing.address or 'Address Not Specified'}
        Location: {listing.city}, {listing.county}, {listing.state}
        Price: ${listing.price:,.0f}
        Acreage: {listing.acreage or 'N/A'}
        Zoning: {listing.zoning.value}
        Source: {listing.source}
        Status: {listing.status.value}
        """
        
        if listing.opportunity_score:
            text += f"\nOpportunity Score: {listing.opportunity_score:.3f}"
        
        if listing.description:
            text += f"\nDescription: {listing.description}"
        
        metadata = {
            "type": "land_listing",
            "city": listing.city,
            "county": listing.county,
            "price": listing.price,
            "acreage": listing.acreage,
            "zoning": listing.zoning.value,
            "url": listing.url
        }
        
        self.add_document(doc_id, text, metadata)
    
    def add_product_optimization(self, product: Any):
        """Add product optimization data to vector database."""
        doc_id = f"product_{product.city}_{product.county}"
        
        text = f"""
        Product Optimization: {product.city}, {product.county}
        Optimal House Size: {product.optimal_sqft_min}-{product.optimal_sqft_max} sqft
        Optimal Bedrooms: {product.optimal_bedrooms}
        Optimal Bathrooms: {product.optimal_bathrooms}
        Average Days on Market: {product.avg_days_on_market}
        """
        
        if product.recommended_features:
            text += "\nRecommended Features:\n"
            for feature in product.recommended_features[:5]:
                text += f"- {feature['feature']} (appears in {feature['frequency']*100:.0f}% of sales)\n"
        
        if product.effective_incentives:
            text += "\nEffective Incentives:\n"
            for incentive in product.effective_incentives:
                text += f"- {incentive['incentive']} (reduces time on market by {incentive['days_on_market_reduction']} days)\n"
        
        metadata = {
            "type": "product_optimization",
            "city": product.city,
            "county": product.county,
            "optimal_sqft_min": product.optimal_sqft_min,
            "optimal_sqft_max": product.optimal_sqft_max
        }
        
        self.add_document(doc_id, text, metadata)
    
    def search(self, query: str, limit: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if not self.qdrant_client:
            logger.warning("Qdrant client not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            
            # Build filter
            query_filter = None
            if filter_type:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.type",
                            match=MatchValue(value=filter_type)
                        )
                    ]
                )
            
            # Search
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "score": result.score
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def query(self, question: str, context_limit: int = 3) -> Dict[str, Any]:
        """Query the system using RAG."""
        logger.info(f"Processing query: {question}")
        
        # Search for relevant context
        relevant_docs = self.search(question, limit=context_limit)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc["text"] for doc in relevant_docs])
        
        # Build prompt
        system_message = """You are an AI assistant specialized in real estate development and market analysis.
        You help users make data-driven decisions about where to build houses, what to build, and how to optimize returns.
        
        Use the provided context to answer questions accurately. If you don't have enough information in the context,
        say so and provide general guidance based on real estate best practices."""
        
        prompt = f"""Context from real estate database:
        {context}
        
        Question: {question}
        
        Please provide a detailed, actionable answer based on the context above."""
        
        # Get response from LLM
        try:
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            sources = [doc["metadata"].get("doc_id", "Unknown") for doc in relevant_docs]
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "context_used": len(relevant_docs),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "context_used": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def bulk_index_data(self, submarkets: List[Any] = None, land_listings: List[Any] = None, products: List[Any] = None):
        """Bulk index all data into vector database."""
        logger.info("Starting bulk indexing")
        
        count = 0
        
        if submarkets:
            for submarket in submarkets:
                self.add_submarket_data(submarket)
                count += 1
        
        if land_listings:
            for listing in land_listings:
                self.add_land_listing(listing)
                count += 1
        
        if products:
            for product in products:
                self.add_product_optimization(product)
                count += 1
        
        logger.info(f"Bulk indexing complete: {count} documents indexed")
    
    def clear_collection(self):
        """Clear all data from the collection."""
        if not self.qdrant_client:
            return
        
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            logger.info("Collection cleared and recreated")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

