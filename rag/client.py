# client.py
# Client script to interact with the Sinhala Letter RAG API

import requests
import json
import sys
import time

API_URL = "http://localhost:8000"  # Change to your GCP VM IP when deployed

def process_query(prompt):
    """Send a query to process and check for missing information."""
    response = requests.post(
        f"{API_URL}/process_query/",
        json={"prompt": prompt}
    )
    
    if response.status_code == 200:
        result = response.json()
        if result["status"] == "incomplete":
            print("Some information is missing. Please provide the following:")
            missing_info = {}
            for field, question in result["questions"].items():
                answer = input(f"{question} ")
                missing_info[field] = answer
            
            # Send another request with the missing information
            return process_query_with_missing_info(prompt, missing_info)
        else:
            return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def process_query_with_missing_info(prompt, missing_info):
    """Send a query with missing information filled in."""
    response = requests.post(
        f"{API_URL}/process_query/",
        json={"prompt": prompt, "missing_info": missing_info}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def generate_letter(original_prompt, enhanced_prompt):
    """Generate a letter using the enhanced prompt."""
    response = requests.post(
        f"{API_URL}/generate_letter/",
        json={"original_prompt": original_prompt, "enhanced_prompt": enhanced_prompt}
    )
    
    if response.status_code == 200:
        return response.json()["generated_letter"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def rate_letter(letter, original_prompt, extracted_info):
    """Ask user to rate the generated letter and submit it to knowledge base if rated highly."""
    print("\n📝 Letter Rating")
    print("=" * 80)
    print("Please rate the quality of this letter on a scale of 1-5:")
    print("1 = Poor | 2 = Fair | 3 = Good | 4 = Very Good | 5 = Excellent")
    
    while True:
        try:
            rating = int(input("Your rating (1-5): "))
            if 1 <= rating <= 5:
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get optional feedback
    feedback = input("Any additional feedback about the letter? (optional): ")
    
    # If rating is high (4 or 5), ask if we can add it to the knowledge base
    if rating >= 4:
        add_to_kb = input("This letter was rated highly! Can we add it to our knowledge base to improve future generations? (y/n): ")
        if add_to_kb.lower() in ["y", "yes"]:
            print("Adding letter to knowledge base...")
            submit_letter_to_kb(letter, original_prompt, extracted_info, rating, feedback)
            print("Letter added successfully! Thank you for your contribution.")
    
    return rating, feedback

def submit_letter_to_kb(letter, original_prompt, extracted_info, rating, feedback):
    """Submit a highly-rated letter to the knowledge base."""
    # Prepare the metadata
    metadata = {
        "subject": extracted_info.get("subject", ""),
        "letter_type": extracted_info.get("letter_type", "general"),
        "rating": rating,
        "feedback": feedback,
        "source": "user_generated",
        "date_added": time.strftime("%Y-%m-%d")
    }
    
    # Submit to the API
    response = requests.post(
        f"{API_URL}/add_to_knowledge_base/",
        json={
            "content": letter,
            "original_prompt": original_prompt,
            "metadata": metadata
        }
    )
    
    if response.status_code != 200:
        print(f"Error adding letter to knowledge base: {response.status_code}")
        print(response.text)

def search_knowledge_base(query, top_k=3):
    """Search the knowledge base directly."""
    response = requests.get(
        f"{API_URL}/search/",
        params={"query": query, "top_k": top_k}
    )
    
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def run_diagnostics():
    """Run diagnostics on the knowledge base."""
    response = requests.get(f"{API_URL}/diagnostics/")
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 Knowledge Base Diagnostics:")
        print("=" * 80)
        print(f"Status: {result['status']}")
        print(f"Document Count: {result['document_count']}")
        print(f"Embedding Model: {result['embedding_model']}")
        
        print("\nSample Documents:")
        for i, doc in enumerate(result.get('sample_documents', []), 1):
            print(f"\n📄 Document {i}:")
            print("-" * 40)
            print(f"ID: {doc.get('id')}")
            print(f"Text: {doc.get('text')[:150]}..." if len(doc.get('text', '')) > 150 else f"Text: {doc.get('text')}")
            print(f"Metadata: {json.dumps(doc.get('metadata', {}), ensure_ascii=False)}")
            print("-" * 40)
        
        test_search = result.get('test_search', {})
        print("\nTest Search Results:")
        print(f"Query: {test_search.get('query')}")
        print(f"Results Found: {test_search.get('results_found')}")
        if test_search.get('first_result'):
            print(f"First Result: {test_search.get('first_result')[:150]}..." 
                  if len(test_search.get('first_result', '')) > 150 
                  else f"First Result: {test_search.get('first_result')}")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def rebuild_knowledge_base():
    """Force rebuild of the knowledge base."""
    print("Rebuilding knowledge base (this may take some time)...")
    response = requests.post(f"{API_URL}/rebuild_knowledge_base/")
    
    if response.status_code == 200:
        result = response.json()
        print("\n🔄 Knowledge Base Rebuild Results:")
        print("=" * 80)
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        print(f"Original data rows: {result['original_data_rows']}")
        print(f"Documents created: {result['documents_created']}")
        print(f"Document chunks: {result['document_chunks']}")
        print(f"Documents in vector store: {result['documents_in_vector_store']}")
        print(f"Time taken: {result['time_taken_seconds']} seconds")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def main():
    """Main function to run the client."""
    print("Sinhala Letter RAG System Client")
    print("--------------------------------")
    
    while True:
        print("\nOptions:")
        print("1. Process letter request")
        print("2. Search knowledge base")
        print("3. Run diagnostics")
        print("4. Rebuild knowledge base")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == "1":
            prompt = input("\nEnter your letter request in Sinhala: ")
            print("\nProcessing your request...")
            
            result = process_query(prompt)
            if result and result["status"] == "complete":
                print("\nExtracted Information:")
                print(json.dumps(result["extracted_info"], indent=2, ensure_ascii=False))
                
                print("\nRelevant Letter Examples Retrieved:")
                for i, doc in enumerate(result["relevant_docs"], 1):
                    print(f"\nExample {i}:")
                    print("-" * 40)
                    print(doc[:200] + "..." if len(doc) > 200 else doc)
                    print("-" * 40)
                
                print("\nGenerating letter...")
                letter = generate_letter(prompt, result["enhanced_prompt"])
                
                if letter:
                    print("\nGenerated Letter:")
                    print("=" * 80)
                    print(letter)
                    print("=" * 80)
                    
                    # Save letter to a file
                    filename = f"generated_letter_{result['extracted_info'].get('letter_type', 'general')}.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(letter)
                    print(f"\nLetter saved to {filename}")
                    
                    # Rate the letter
                    # rating, feedback = rate_letter(letter, prompt, result["extracted_info"])
                    # print(f"\nThank you for rating the letter ({rating}/5)!")
                    
                    # # Provide additional feedback based on rating
                    # if rating <= 2:
                    #     print("We appreciate your feedback and will work on improving our letter generation.")
                    # elif rating == 3:
                    #     print("Thank you for your feedback. We'll continue to refine our system.")
                    # else:
                    #     print("We're glad you found the letter helpful!")
            
        elif choice == "2":
            query = input("\nEnter your search query: ")
            top_k = int(input("Enter number of results to retrieve (default 3): ") or "3")
            
            print("\nSearching knowledge base...")
            results = search_knowledge_base(query, top_k)
            
            if results:
                print(f"\nFound {len(results)} relevant documents:")
                for i, doc in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print("-" * 40)
                    print(f"Content: {doc['content'][:200]}..." if len(doc['content']) > 200 else f"Content: {doc['content']}")
                    print(f"Subject: {doc['metadata'].get('subject', 'N/A')}")
                    print(f"Tags: {', '.join(doc['metadata'].get('tags', []))}")
                    print("-" * 40)
        
        elif choice == "3":
            print("\nRunning system diagnostics...")
            run_diagnostics()
        
        elif choice == "4":
            rebuild_knowledge_base()
        
        elif choice == "5":
            print("\nExiting. Thank you for using the Sinhala Letter RAG System!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()