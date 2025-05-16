"""
Simple demonstration of text summarization using Ant Colony Optimization.

This script loads sample documents and applies ACO summarization to generate concise summaries.
"""

import time
from data.sample_documents import DOCUMENTS
from aco_summarizer import ACOSummarizer


def print_document_info(doc):
    """Print information about a document."""
    print(f"\n{'=' * 60}")
    print(f"DOCUMENT: {doc['title']} (ID: {doc['id']})")
    print(f"{'-' * 60}")
    print(f"ORIGINAL TEXT ({len(doc['text'].split())} words):")
    print(f"{doc['text']}")
    print(f"\nHUMAN SUMMARY ({len(doc['summary'].split())} words):")
    print(f"{doc['summary']}")
    print(f"{'-' * 60}")


def print_summary_result(doc, aco_summary, selected_indices):
    """Print the ACO-generated summary."""
    print(f"ACO SUMMARY ({len(aco_summary.split())} words):")
    print(f"{aco_summary}")
    print(f"\nSelected sentences: {selected_indices}")
    
    # Calculate compression ratio
    original_words = len(doc['text'].split())
    summary_words = len(aco_summary.split())
    compression = summary_words / original_words * 100
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"- Original length: {original_words} words")
    print(f"- Summary length: {summary_words} words")
    print(f"- Compression ratio: {compression:.1f}%")
    print(f"{'=' * 60}\n")





def main():
    """Main function to run the ACO summarization demo."""
    print("\nTEXT SUMMARIZATION USING ANT COLONY OPTIMIZATION")
    print("=" * 60)
    print("A simple demonstration for university discussion")
    print("-" * 60)
    
    # Explain the ACO parameters
    print("\nACO PARAMETERS:")
    print("- num_ants: 10 (Number of ants in colony)")
    print("- alpha: 1.0 (Importance of pheromone trails)")
    print("- beta: 2.0 (Importance of sentence scores)")
    print("- rho: 0.1 (Pheromone evaporation rate)")
    print("- max_iterations: 30 (Number of iterations)")
    print("- compression_ratio: 0.3 (Target summary length)")
    
    # Initialize summarizer with simplified parameters
    summarizer = ACOSummarizer(
        num_ants=10,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q0=0.9,
        max_iterations=30,
        compression_ratio=0.3
    )
    
    # Process each document
    for doc in DOCUMENTS:
        print_document_info(doc)
        
        # Generate summary using ACO
        print("Generating summary using ACO...")
        start_time = time.time()
        aco_summary, selected_indices = summarizer.summarize(doc['text'])
        end_time = time.time()
        
        print(f"Summary generated in {end_time - start_time:.2f} seconds")
        
        # Print results
        print_summary_result(doc, aco_summary, selected_indices)
        
        # Add a separator between documents
        if doc != DOCUMENTS[-1]:
            print("\n" + "-" * 60 + "\n")
    
    print("\nSUMMARY GENERATION COMPLETE!")
  

if __name__ == "__main__":
    main()
