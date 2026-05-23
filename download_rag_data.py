#!/usr/bin/env python3
"""
RAG Data Collection Script
Download QA pairs from SQUAD and documents from Wikipedia
Ready to use for RAG system testing
"""

import json
import os
from pathlib import Path

def download_squad_dataset(num_samples=100, output_file="squad_qa.json"):
    """
    Download SQUAD dataset for QA pairs
    
    Args:
        num_samples: Number of QA pairs to download (max 88k)
        output_file: Where to save the data
    """
    print(f"📥 Downloading SQUAD dataset ({num_samples} samples)...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install: pip install datasets")
        return False
    
    try:
        # Load SQUAD
        print("   Loading SQUAD...")
        squad = load_dataset("squad", split=f"train[:{num_samples}]")
        
        # Convert to RAG format
        qa_pairs = []
        for idx, item in enumerate(squad):
            qa_pairs.append({
                'id': f"squad_{idx}",
                'query': item['question'],
                'document': item['context'],
                'answer': item['answers']['text'][0] if item['answers']['text'] else "N/A",
                'source': 'SQUAD'
            })
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Downloaded {len(qa_pairs)} QA pairs")
        print(f"📁 Saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading SQUAD: {e}")
        return False


def download_wikipedia_documents(topics=None, output_file="wiki_documents.json"):
    """
    Download Wikipedia articles as documents
    
    Args:
        topics: List of Wikipedia topics to download
        output_file: Where to save the documents
    """
    
    if topics is None:
        topics = [
            "Machine Learning",
            "Natural Language Processing",
            "Deep Learning",
            "Artificial Intelligence",
            "Data Science",
            "Neural Network",
            "Transformer (machine learning)",
            "Retrieval-augmented generation",
            "Vector database",
            "Semantic search",
        ]
    
    print(f"\n📥 Downloading Wikipedia documents ({len(topics)} topics)...")
    
    try:
        import wikipediaapi
    except ImportError:
        print("❌ Please install: pip install wikipedia-api")
        return False
    
    try:
        wiki = wikipediaapi.Wikipedia('en')
        documents = []
        
        for topic in topics:
            print(f"   Downloading: {topic}...", end=" ")
            page = wiki.page(topic)
            
            if page.exists():
                documents.append({
                    'id': topic.lower().replace(' ', '_'),
                    'title': topic,
                    'content': page.text,
                    'url': page.fullurl,
                    'length': len(page.text)
                })
                print(f"✓ ({len(page.text)} chars)")
            else:
                print("✗ Not found")
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Downloaded {len(documents)} Wikipedia documents")
        print(f"📁 Saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading Wikipedia: {e}")
        return False


def download_arxiv_papers(query="retrieval augmented generation", max_results=20, output_file="arxiv_papers.json"):
    """
    Download ArXiv papers
    
    Args:
        query: Search query
        max_results: Number of papers to download
        output_file: Where to save
    """
    print(f"\n📥 Searching ArXiv for: '{query}' ({max_results} papers)...")
    
    try:
        import arxiv
    except ImportError:
        print("❌ Please install: pip install arxiv")
        return False
    
    try:
        client = arxiv.Client()
        papers = []
        
        results = client.results(
            arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
        )
        
        for idx, result in enumerate(results):
            print(f"   {idx+1}. {result.title[:60]}...", end=" ")
            
            papers.append({
                'id': result.entry_id.split('/abs/')[-1],
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'published': str(result.published),
                'url': result.entry_id,
                'pdf_url': result.pdf_url
            })
            print("✓")
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Downloaded {len(papers)} ArXiv papers")
        print(f"📁 Saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading ArXiv: {e}")
        return False


def combine_datasets(qa_file, documents_file, output_file="rag_dataset.json"):
    """
    Combine QA pairs and documents into single dataset
    """
    print(f"\n🔗 Combining datasets...")
    
    try:
        # Load data
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Combine
        dataset = {
            'metadata': {
                'qa_pairs_count': len(qa_pairs),
                'documents_count': len(documents),
                'total_documents_chars': sum(len(d.get('content', '')) or len(d.get('summary', '')) for d in documents)
            },
            'qa_pairs': qa_pairs,
            'documents': documents
        }
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Combined dataset created")
        print(f"   - QA pairs: {len(qa_pairs)}")
        print(f"   - Documents: {len(documents)}")
        print(f"📁 Saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error combining: {e}")
        return False


def verify_dataset(qa_file, documents_file):
    """
    Verify that data is properly formatted
    """
    print(f"\n✅ Verifying datasets...")
    
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   QA Pairs: {len(qa_data)}")
        if qa_data:
            print(f"   - Sample query: {qa_data[0]['query'][:80]}...")
            print(f"   - Sample answer: {qa_data[0]['answer'][:80]}...")
        
        print(f"\n   Documents: {len(doc_data)}")
        if doc_data:
            doc = doc_data[0]
            title = doc.get('title') or doc.get('id')
            print(f"   - Sample document: {title}")
            content = doc.get('content') or doc.get('summary', '')
            print(f"   - Length: {len(content)} characters")
        
        print(f"\n✅ All data looks good! Ready for RAG testing.")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """
    Main script: Download all data
    """
    print("=" * 60)
    print("RAG DATA COLLECTION SCRIPT")
    print("=" * 60)
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    os.chdir("data")
    
    # Download options
    print("\nWhich data would you like to download?")
    print("1. SQUAD + Wikipedia (Quick start) ⭐ RECOMMENDED")
    print("2. SQUAD only (Minimal)")
    print("3. Wikipedia only")
    print("4. ArXiv papers")
    print("5. All of the above")
    
    choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"
    
    results = {}
    
    if choice in ["1", "5"]:
        print("\n" + "="*60)
        print("STEP 1: SQUAD Dataset")
        print("="*60)
        results['squad'] = download_squad_dataset(num_samples=100, output_file="squad_qa.json")
    
    if choice in ["1", "3", "5"]:
        print("\n" + "="*60)
        print("STEP 2: Wikipedia Documents")
        print("="*60)
        results['wiki'] = download_wikipedia_documents(output_file="wiki_documents.json")
    
    if choice in ["4", "5"]:
        print("\n" + "="*60)
        print("STEP 3: ArXiv Papers")
        print("="*60)
        results['arxiv'] = download_arxiv_papers(output_file="arxiv_papers.json")
    
    if choice in ["2"]:
        print("\n" + "="*60)
        print("STEP 1: SQUAD Dataset Only")
        print("="*60)
        results['squad'] = download_squad_dataset(num_samples=100, output_file="squad_qa.json")
    
    # Combine if we have both QA and documents
    print("\n" + "="*60)
    print("FINAL STEP: Combining datasets")
    print("="*60)
    
    if choice in ["1", "5"]:
        combine_datasets("squad_qa.json", "wiki_documents.json", "rag_dataset.json")
        verify_dataset("squad_qa.json", "wiki_documents.json")
    elif choice in ["3"]:
        verify_dataset("squad_qa.json", "wiki_documents.json")
    
    print("\n" + "="*60)
    print("✅ DATA COLLECTION COMPLETE!")
    print("="*60)
    print("\n📁 Files created in ./data/")
    print("\n🚀 Next steps:")
    print("1. Review the downloaded data")
    print("2. Proceed to Week 1 of the 45-day RAG plan")
    print("3. Build your baseline RAG system")
    print("\nHappy RAG building! 🎉")


if __name__ == "__main__":
    main()
