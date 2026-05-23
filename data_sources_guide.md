# 📊 HƯỚNG DẪN TÌM DỮ LIỆU CHO THỰC NGHIỆM RAG

## OVERVIEW: DỮ LIỆU BẠN CẦN LÀ GÌ?

Cho project RAG Precision, bạn cần **2 loại dữ liệu**:

```
1. DOCUMENTS (Tài liệu)
   - Là những text mà RAG sẽ search qua
   - Ví dụ: technical docs, FAQs, wiki, blog posts
   - Mục tiêu: 50-100 documents, mỗi document 500-2000 words
   - Format: PDF, TXT, HTML, Markdown

2. TEST QUERIES + EXPECTED ANSWERS (QA pairs)
   - Là những câu hỏi bạn sẽ test
   - Cần ground truth answers để evaluate
   - Mục tiêu: 100 QA pairs (70 train, 30 test)
   - Format: JSON, CSV, hoặc Excel
```

---

## 🟢 OPTION 1: FREE PUBLIC DATASETS (Recommended for quick start)

### 1.1 **Technical Documentation** (Easiest to start)

#### A. Wikipedia + ArXiv (Most accessible)

**Wikipedia:**
```python
# Download using wikipedia-api
pip install wikipedia-api

import wikipediaapi
wiki = wikipediaapi.Wikipedia('en')

# Get full articles
page = wiki.page('Machine_Learning')
if page.exists():
    text = page.text
    # Save to file
    with open('ml_doc.txt', 'w') as f:
        f.write(text)
```

**Best Wikipedia topics for RAG testing:**
- Machine Learning
- Natural Language Processing  
- Deep Learning
- Computer Science
- Artificial Intelligence
- Data Science
- Programming Languages

**ArXiv (Academic papers):**
```python
# Download using arxiv-api
pip install arxiv

import arxiv

# Search RAG papers
client = arxiv.Client()
results = client.results(
    arxiv.Search(
        query='cat:cs.CL AND (RAG OR "retrieval augmented")',
        max_results=20
    )
)

for result in results:
    print(result.title)
    print(result.summary)
    # Download PDF
    result.download_pdf(dirpath="./papers")
```

**Steps:**
```
1. Go to https://arxiv.org/
2. Search: "retrieval augmented generation" or "semantic search"
3. Download PDFs
4. Extract text using pdfplumber or PyPDF2

pip install pdfplumber
import pdfplumber

with pdfplumber.open('paper.pdf') as pdf:
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
```

**Time needed:** 2-3 hours for 50 documents  
**Quality:** ⭐⭐⭐⭐⭐ (High quality, well-written)  
**Cost:** FREE ✅

---

#### B. GitHub README + Documentation (Domain-specific)

**Target projects:**
- LlamaIndex repo (RAG framework docs)
- LangChain docs
- OpenAI cookbook
- Hugging Face documentation

**How to get:**
```bash
# Clone repository
git clone https://github.com/run-llm/llama_index.git

# Extract markdown files
find . -name "*.md" | head -50 > docs.txt

# Or download directly from GitHub web interface
# Go to: https://github.com/run-llm/llama_index/tree/main/docs
```

**Time needed:** 1-2 hours  
**Quality:** ⭐⭐⭐⭐⭐ (Technical, relevant)  
**Cost:** FREE ✅

---

#### C. Stack Overflow + Dev.to (Q&A format - PERFECT for RAG)

**Stack Overflow Dataset:**
```bash
# Download Stack Overflow dump (most recent: Dec 2024)
# Go to: https://archive.org/details/stackexchange

# Or use Stack Overflow API (limited but free)
curl "https://api.stackexchange.com/2.3/questions?site=stackoverflow&tagged=python&sort=votes&order=desc" > so_data.json
```

**Dev.to API (easier):**
```python
import requests
import json

# Get top articles
url = "https://dev.to/api/articles?top=100"
response = requests.get(url)
articles = response.json()

# Save
with open('devto_articles.json', 'w') as f:
    json.dump(articles, f)

# Extract text
for article in articles:
    title = article['title']
    body = article['body_markdown']
    print(f"Title: {title}")
    print(f"Content: {body[:200]}")
```

**Time needed:** 1 hour  
**Quality:** ⭐⭐⭐⭐ (Q&A format, diverse)  
**Cost:** FREE ✅

---

### 1.2 **Ready-made QA Datasets** (Best for evaluation)

#### A. SQUAD v2 (Stanford Question Answering Dataset)

```python
# Use Hugging Face datasets library
from datasets import load_dataset

# Load SQUAD
squad = load_dataset("squad")
train_data = squad['train']
validation_data = squad['validation']

# Format: {context, question, answers}
example = train_data[0]
print(example)
# Output:
# {
#   'id': 'string',
#   'title': 'Wikipedia article title',
#   'context': 'long paragraph',
#   'question': 'what is...',
#   'answers': {
#       'text': ['answer1', 'answer2'],
#       'answer_start': [123, 456]
#   }
# }

# Convert to RAG format
qa_pairs = []
for item in train_data[:100]:  # First 100
    qa_pairs.append({
        'query': item['question'],
        'documents': [item['context']],
        'ground_truth': item['answers']['text'][0]
    })

import json
with open('squad_qa.json', 'w') as f:
    json.dump(qa_pairs, f)
```

**Advantages:**
- ✅ Already in Q&A format
- ✅ Has ground truth answers
- ✅ 100k+ questions available
- ✅ Can download directly

**Code:**
```bash
pip install datasets

# Python script
from datasets import load_dataset
squad = load_dataset("squad", split="train[:1000]")
# Now you have 1000 QA pairs ready
```

**Time needed:** 30 minutes  
**Quality:** ⭐⭐⭐⭐⭐ (Gold standard)  
**Cost:** FREE ✅

---

#### B. MS MARCO (Microsoft Machine Reading Comprehension)

```python
# Similar to SQUAD but bigger
from datasets import load_dataset

ms_marco = load_dataset("ms_marco", "v2.1")

# Format is similar to SQUAD
for example in ms_marco['train'].take(10):
    print(example['query'])
    print(example['passages'])
    print(example['answers'])
```

**Time needed:** 30 minutes  
**Quality:** ⭐⭐⭐⭐⭐  
**Cost:** FREE ✅

---

#### C. Natural Questions Dataset (Google)

```python
from datasets import load_dataset

nq = load_dataset("natural_questions")

# Has both short and long answers
for example in nq['train'].take(10):
    print(example['question'])
    print(example['document']['text'])
    print(example['short_answers'])
```

**Time needed:** 30 minutes  
**Quality:** ⭐⭐⭐⭐⭐  
**Cost:** FREE ✅

---

### 1.3 **Domain-Specific Datasets** (Choose based on your interest)

| Domain | Dataset | Source | Size | Format |
|--------|---------|--------|------|--------|
| **Medical** | MEDIQA | https://huggingface.co/datasets/MEDIQA | 188k | QA |
| **Legal** | Legal-BERT | https://huggingface.co/nlpaueb/legal-bert | Various | Text |
| **Scientific** | PubMed | https://pubmed.ncbi.nlm.nih.gov/ | 35M+ | Abstracts |
| **Programming** | CodeSearchNet | https://github.com/github/CodeSearchNet | 6M+ | Code |
| **News** | News QA | https://huggingface.co/datasets/newsqa | 100k+ | QA |
| **Finance** | FinQA | https://huggingface.co/datasets/FinQA | 13k | QA |

**How to load:**
```python
from datasets import load_dataset

# Any dataset from Hugging Face
dataset = load_dataset("mediqa")
dataset = load_dataset("newsqa")
dataset = load_dataset("financial_phrasebank")
```

**Time needed:** 30 minutes per dataset  
**Cost:** FREE ✅

---

## 🟡 OPTION 2: SYNTHETIC DATA (CREATE YOUR OWN)

### Why use synthetic data?

✅ Full control over domain  
✅ Guaranteed relevance to your problem  
✅ Can test specific edge cases  
✅ Domain-specific (if you have specific use case)  

**Tradeoff:** Takes longer but higher quality

---

### 2.1 **Generate using Claude API** (Recommended)

**Strategy: Create domain-specific dataset**

```python
from anthropic import Anthropic

client = Anthropic()

def generate_qa_pairs(topic, num_pairs=10):
    """Generate QA pairs for a specific topic using Claude"""
    
    conversation_history = []
    
    # Step 1: Ask Claude to write an article
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Write a comprehensive technical article (1500 words) about: {topic}
            
            Make it detailed, informative, and suitable for creating QA pairs from it."""
        }]
    )
    
    article = message.content[0].text
    
    # Step 2: Generate QA pairs from the article
    qa_prompt = f"""Based on this article:

{article}

Generate {num_pairs} question-answer pairs. Each pair should:
1. Have a clear, specific question
2. Have an answer that can be found in the article
3. Be diverse (some factual, some conceptual, some analytical)

Format as JSON array:
[
  {{"query": "question?", "answer": "answer", "context": "relevant section"}},
  ...
]"""

    qa_message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": qa_prompt
        }]
    )
    
    qa_text = qa_message.content[0].text
    
    # Parse JSON
    import json
    import re
    
    json_match = re.search(r'\[.*\]', qa_text, re.DOTALL)
    if json_match:
        qa_pairs = json.loads(json_match.group())
    else:
        qa_pairs = []
    
    return {
        'document': article,
        'qa_pairs': qa_pairs
    }

# Generate multiple topics
topics = [
    "Retrieval Augmented Generation (RAG) Architecture",
    "Vector Databases and Semantic Search",
    "Prompt Engineering Best Practices",
    "LLM Fine-tuning Techniques",
    "Multi-agent Systems in AI"
]

all_data = []
for topic in topics:
    print(f"Generating data for: {topic}")
    data = generate_qa_pairs(topic, num_pairs=20)
    all_data.append(data)
    
# Save
import json
with open('synthetic_dataset.json', 'w') as f:
    json.dump(all_data, f, indent=2)

print(f"Generated {len(all_data)} documents with ~100 QA pairs total")
```

**Cost:** ~$2-5 USD (depending on document length)  
**Time:** 2-3 hours  
**Quality:** ⭐⭐⭐⭐⭐ (Domain-specific, high quality)  
**Control:** ✅ Full control

---

### 2.2 **Use Open Source Models Locally** (Free but slower)

```python
# Option A: Use Ollama (free, local)
# Install: https://ollama.ai/
# Pull model: ollama pull mistral

import requests
import json

def generate_with_ollama(prompt):
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'mistral',
            'prompt': prompt,
            'stream': False
        }
    )
    return response.json()['response']

# Generate documents
prompt = """Write a technical article about vector databases.
Make it 1000-1500 words, detailed and informative."""

article = generate_with_ollama(prompt)
print(article)
```

**Cost:** FREE (but uses your computer)  
**Time:** 30 min - 2 hours (slow on CPU)  
**Quality:** ⭐⭐⭐ (Acceptable but lower than Claude)

---

### 2.3 **Template-based Synthetic Data** (Fastest)

```python
import json
import random

# Define templates
questions_template = {
    "What": [
        "What is {topic}?",
        "What are the benefits of {topic}?",
        "What are the main components of {topic}?",
    ],
    "How": [
        "How does {topic} work?",
        "How to implement {topic}?",
        "How to optimize {topic}?",
    ],
    "Why": [
        "Why is {topic} important?",
        "Why should we use {topic}?",
    ],
    "Compare": [
        "Compare {topic1} vs {topic2}",
        "What's the difference between {topic1} and {topic2}?",
    ]
}

topics = [
    "Retrieval Augmented Generation",
    "Semantic Search",
    "Vector Embeddings",
    "LLM Fine-tuning",
    "Prompt Engineering"
]

qa_pairs = []
for question_type, templates in questions_template.items():
    for template in templates:
        for topic in topics:
            if "{topic1}" in template:
                continue  # Skip for now
            
            query = template.format(topic=topic)
            # Generate synthetic but reasonable answer
            answer = f"[Answer about {topic}. This is placeholder for {question_type} question.]"
            
            qa_pairs.append({
                'query': query,
                'answer': answer,
                'type': question_type
            })

print(f"Generated {len(qa_pairs)} QA pairs")
# Save to file
with open('template_qa.json', 'w') as f:
    json.dump(qa_pairs, f)
```

**Cost:** FREE  
**Time:** 30 minutes  
**Quality:** ⭐⭐ (Quick but needs real answers)  
**Use case:** For quick testing, not final evaluation

---

## 🔴 OPTION 3: COMPANY'S OWN DATA (Best for real value)

**If you can access:**
- Internal documentation
- Internal FAQs
- Internal knowledge base
- Internal chat logs

**This is IDEAL because:**
✅ Domain-specific to your company  
✅ Real-world use case  
✅ Show actual ROI in presentation  

**Steps:**
```
1. Get permission from company
2. Request data dump (format: PDF, HTML, Markdown)
3. Anonymize if needed (remove customer names, etc.)
4. Use as your document corpus
5. Create test queries based on employee usage patterns
6. Perfect for demo: "This is your actual internal data"
```

---

## 🎯 MY RECOMMENDATION: HYBRID APPROACH

**For fastest, highest-quality startup:**

### **Phase 1: Week 1 (Quick start with free data)**
```
- Use SQUAD dataset for QA pairs (30 min)
  → 100 Q&A pairs, ground truth answers ready
  
- Use Wikipedia + ArXiv for documents (2 hours)
  → 50-100 technical documents
  
- Total time: 3 hours
- Quality: ⭐⭐⭐⭐⭐
- Cost: FREE
```

**Deliverable:** Full RAG system that works, evaluation pipeline ready

---

### **Phase 2: Week 2-3 (Enhance with domain-specific)**
```
- Generate synthetic data with Claude (2-3 hours)
  Topics:
  - Retrieval Augmented Generation
  - Vector Databases
  - Semantic Search
  - LLM Optimization
  - Prompt Engineering
  
- Total additional documents: 5-10
- Additional QA pairs: 100-200
```

**Deliverable:** More comprehensive dataset, better evaluation

---

### **Phase 3: Week 4 (Real data if possible)**
```
- If company provides internal docs:
  → Use as "production" test set
  → Show real metrics on real company data
  
- If not available:
  → Stick with synthetic + public data
  → Still valid for research/demo
```

**Deliverable:** Production-ready evaluation

---

## 📋 STEP-BY-STEP: START TODAY

### **Option A: Fastest Path (Recommended) - 3 hours**

**Hour 1: Get SQUAD dataset**
```bash
# Python script
from datasets import load_dataset
import json

squad = load_dataset("squad", split="train[:100]")

qa_data = []
for item in squad:
    qa_data.append({
        'query': item['question'],
        'document': item['context'],
        'answer': item['answers']['text'][0]
    })

with open('squad_100_qa.json', 'w') as f:
    json.dump(qa_data, f)

print(f"Saved {len(qa_data)} QA pairs")
```

**Hour 2-3: Get Wikipedia documents**
```bash
# Using Wikipedia API
pip install wikipedia-api pdfplumber

# Python script
import wikipediaapi
import json

wiki = wikipediaapi.Wikipedia('en')

topics = [
    "Machine Learning",
    "Natural Language Processing",
    "Deep Learning",
    "Artificial Intelligence",
    "Data Science",
    "Neural Network",
    "Transformer Architecture"
]

documents = []
for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        documents.append({
            'title': topic,
            'content': page.text
        })

with open('wiki_documents.json', 'w') as f:
    json.dump(documents, f)

print(f"Saved {len(documents)} Wikipedia documents")
```

**Result:** 
- ✅ 100 QA pairs with ground truth
- ✅ 7 high-quality technical documents
- ✅ Ready to build RAG system
- ✅ Cost: FREE
- ✅ Time: 3 hours

---

### **Option B: Domain-Specific Path (Better quality) - 6 hours**

**Hour 1:** Get base datasets (same as Option A)

**Hour 2-4:** Generate with Claude
```python
# See code example above
# Generate documents about:
# - RAG systems
# - Vector search
# - Semantic embeddings
# - Prompt optimization

# Cost: ~$3-5 USD
# Output: 5 domain-specific documents + 100-200 QA pairs
```

**Hour 5-6:** Create test queries manually
```python
# Hand-craft 20-30 tough test queries
# Examples:
# - "Compare BM25 vs semantic search"
# - "What are the failure modes of RAG?"
# - "How to optimize retrieval latency?"
# - "When should you use RAG vs fine-tuning?"

# This ensures you test the right things
```

**Result:**
- ✅ 100-300 QA pairs
- ✅ 12-15 high-quality documents
- ✅ Domain-specific test cases
- ✅ Cost: $3-5 USD
- ✅ Time: 6 hours

---

## 🚀 WHAT TO DO RIGHT NOW (Next 30 minutes)

**Choose ONE:**

### **Choice 1: Go Fast (Recommend)**
```bash
# Install
pip install datasets

# Run this Python script
from datasets import load_dataset
squad = load_dataset("squad", split="train[:100]")
print(f"Loaded {len(squad)} QA pairs")

# Now you have data ready
```

**Then:** Proceed to build baseline RAG with this data

---

### **Choice 2: Get Better Data**
```bash
# If you prefer domain-specific data
# Message company: "Need sample internal docs for RAG testing"
# They provide: PDFs, internal wiki, or documentation

# While waiting:
# Download Wikipedia articles (see code above)
```

---

### **Choice 3: Generate Synthetic**
```python
# Using Claude API to generate perfect data
# Cost: ~$3-5
# Time: 2-3 hours
# Quality: Highest for your domain
```

---

## 📊 COMPARISON TABLE: ALL OPTIONS

| Source | Effort | Quality | Cost | Time | Best For |
|--------|--------|---------|------|------|----------|
| **SQUAD** | 30 min | ⭐⭐⭐⭐⭐ | FREE | 30 min | Quick start ✅ |
| **Wikipedia** | 1-2 hrs | ⭐⭐⭐⭐ | FREE | 1-2 hrs | General purpose |
| **ArXiv** | 2-3 hrs | ⭐⭐⭐⭐⭐ | FREE | 2-3 hrs | Research papers |
| **Stack Overflow** | 1-2 hrs | ⭐⭐⭐⭐ | FREE | 1-2 hrs | Q&A format |
| **Claude-generated** | 2-3 hrs | ⭐⭐⭐⭐⭐ | $3-5 | 2-3 hrs | Domain-specific |
| **Company internal** | Variable | ⭐⭐⭐⭐⭐ | FREE | Variable | Production test |
| **MS MARCO** | 30 min | ⭐⭐⭐⭐⭐ | FREE | 30 min | Large scale |

---

## ✅ FINAL CHECKLIST

Before you start building RAG:

- [ ] Download SQUAD dataset (QA pairs) OR choose alternative
- [ ] Get 50-100 documents (Wikipedia, ArXiv, or company docs)
- [ ] Convert to format: `{query, document, answer}`
- [ ] Save as JSON in your project folder
- [ ] Create simple notebook to explore data

```python
# Verification script
import json

# Load data
with open('squad_qa.json') as f:
    qa_data = json.load(f)

with open('documents.json') as f:
    docs = json.load(f)

print(f"✅ QA pairs: {len(qa_data)}")
print(f"✅ Documents: {len(docs)}")
print(f"✅ Ready to build RAG!")

# Show sample
print("\nSample QA pair:")
print(qa_data[0])
```

**Once you have data → proceed to Week 1 of the 45-day plan!**

---

## 📞 QUESTIONS ANSWERED

**Q: Cần bao nhiêu data?**  
A: 50-100 documents + 100-200 QA pairs là đủ để test RAG techniques

**Q: Format nào tốt nhất?**  
A: JSON với `{query, documents[], ground_truth_answer}` format

**Q: Bao lâu để download?**  
A: SQUAD + Wikipedia = 3 hours total

**Q: Có thể dùng production data không?**  
A: ✅ Yes! Đó là ideal choice. Nhưng public data cũng ok để start.

**Q: Cần phải tạo synthetic data không?**  
A: Not required, but recommended for domain-specific accuracy

**Q: Budget cho dữ liệu?**  
A: Mostly FREE (Wikipedia, SQUAD, etc). Optional: $3-5 for Claude generation

---

**Ready? Pick Option A (SQUAD + Wikipedia) and let's go! 🚀**
