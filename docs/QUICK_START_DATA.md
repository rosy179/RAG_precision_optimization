# ⚡ QUICK START: GET RAG DATA TODAY (30 MINUTES)

## OPTION 1: Automatic Script (Recommended) ⭐

```bash
# 1. Copy script to your project
wget https://your-repo/download_rag_data.py

# 2. Install dependencies
pip install datasets wikipedia-api arxiv

# 3. Run script
python scripts/download_rag_data.py

# 4. Choose option 1 (SQUAD + Wikipedia)
# Answer: 1

# 5. Wait 10-15 minutes
# Script downloads everything automatically

# 6. Check results
ls -la data/
# You'll see:
# - squad_qa.json (100 QA pairs)
# - wiki_documents.json (10 Wikipedia articles)
# - rag_dataset.json (combined)
```

**Time: 15 minutes** ✅  
**Cost: FREE** ✅  
**Quality: ⭐⭐⭐⭐⭐** ✅  

---

## OPTION 2: Manual Download (If script fails)

### Step 1: Install libraries (2 min)
```bash
pip install datasets wikipedia-api arxiv pdfplumber
```

### Step 2: Download SQUAD (5 min)
```python
# save as get_squad.py
from datasets import load_dataset
import json

squad = load_dataset("squad", split="train[:100]")

qa_pairs = []
for item in squad:
    qa_pairs.append({
        'query': item['question'],
        'document': item['context'],
        'answer': item['answers']['text'][0]
    })

with open('squad_qa.json', 'w') as f:
    json.dump(qa_pairs, f)

print(f"✅ Downloaded {len(qa_pairs)} QA pairs")
```

**Run:**
```bash
python get_squad.py
```

### Step 3: Download Wikipedia (5 min)
```python
# save as get_wiki.py
import wikipediaapi
import json

wiki = wikipediaapi.Wikipedia('en')

topics = [
    "Machine Learning",
    "Natural Language Processing",
    "Deep Learning",
    "Artificial Intelligence",
    "Retrieval-augmented generation",
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

print(f"✅ Downloaded {len(documents)} documents")
```

**Run:**
```bash
python get_wiki.py
```

### Step 4: Verify (2 min)
```python
# save as verify_data.py
import json

with open('squad_qa.json') as f:
    qa = json.load(f)

with open('wiki_documents.json') as f:
    docs = json.load(f)

print(f"✅ QA pairs: {len(qa)}")
print(f"✅ Documents: {len(docs)}")
print(f"✅ Ready for RAG!")
```

**Run:**
```bash
python verify_data.py
```

---

## OPTION 3: Use Pre-existing Datasets (Instant)

If you just want to test immediately without waiting:

```python
# Download HuggingFace datasets (no waiting)
from datasets import load_dataset

# Option A: MS MARCO (larger)
marco = load_dataset("ms_marco", "v2.1", split="train[:1000]")

# Option B: Natural Questions
nq = load_dataset("natural_questions", split="train[:1000]")

# Option C: NewsQA  
newsqa = load_dataset("newsqa", split="train[:1000]")

# All of these have Q&A format ready to use
```

---

## OPTION 4: Company Data (Best but requires permission)

If you can get internal company data:

```
1. Email company:
   "Hi, for RAG testing, can you share:
   - Internal documentation (PDF/HTML)
   - Internal FAQs
   - Knowledge base articles
   - Technical docs"

2. They send you files

3. Convert to JSON:
   {
     'documents': [
       {'title': '...', 'content': '...'},
       ...
     ]
   }

4. Create test queries based on your knowledge
   
5. Now you have REAL data for testing
```

---

## 🎯 WHAT TO DO RIGHT NOW

### **Pick ONE path (don't overthink):**

#### **Path A: Fastest (15 min) ⭐ RECOMMENDED**
```bash
python scripts/download_rag_data.py
# Answer: 1
# Wait 10 minutes
# Done!
```

#### **Path B: Manual (20 min)**
```bash
python get_squad.py
python get_wiki.py
python verify_data.py
```

#### **Path C: Super Fast (5 min, requires internet)**
```python
from datasets import load_dataset
squad = load_dataset("squad", split="train[:100]")
# Use immediately
```

#### **Path D: Company Data (variable)**
```
Ask company for internal docs
While waiting, use Path A or B as backup
```

---

## ✅ DATA STRUCTURE YOU'll GET

After any path above, you'll have data in this format:

```json
{
  "qa_pairs": [
    {
      "query": "What is machine learning?",
      "document": "Machine learning is a subset of artificial intelligence...",
      "answer": "A subset of AI that learns from data",
      "source": "SQUAD"
    },
    ...
  ],
  "documents": [
    {
      "title": "Machine Learning",
      "content": "Machine learning is a type of data analysis...",
      "url": "https://en.wikipedia.org/wiki/Machine_learning"
    },
    ...
  ]
}
```

**This format is perfect for RAG testing! ✅**

---

## 📊 EXPECTED OUTPUT

After completing any path:

```
✅ 100 QA pairs (with ground truth answers)
✅ 10-20 high-quality documents
✅ Ready to build RAG system
✅ Ready to evaluate with Ragas
✅ Cost: FREE
✅ Time: 15-30 minutes
```

---

## 🚫 TROUBLESHOOTING

### Problem: "datasets module not found"
```bash
pip install datasets
```

### Problem: "wikipedia-api timeout"
```bash
# Just retry, or
# Skip Wikipedia and use SQUAD only
```

### Problem: "JSON decode error"
```bash
# Check file encoding
# Add encoding='utf-8' when saving
```

### Problem: "Not enough disk space"
```bash
# SQUAD + Wikipedia = ~50MB max
# Should be fine on any modern computer
```

---

## 📞 NEED HELP?

### If you can't download:
```
Option 1: Use this pre-prepared dataset
curl https://huggingface.co/datasets/squad/raw/main/data/train.json

Option 2: Create synthetic data
Use Claude API to generate test data ($3-5)

Option 3: Wait for company data
Continue with manual testing meanwhile
```

---

## 🎉 AFTER DATA IS READY

1. ✅ Verify you have data files
2. ✅ Check formats match expected structure  
3. ✅ Proceed to Week 1 of 45-day plan
4. ✅ Build baseline RAG system
5. ✅ Start implementing techniques

---

**STEP 1 (Right now): Download data - 30 min**
**STEP 2 (Tomorrow): Setup RAG - 4 hours**  
**STEP 3 (Next week): Implement techniques - Week 2-3**  

**Let's go! 🚀**

---

## 💡 BONUS: For the impatient

Absolute minimum to start coding TODAY:

```python
# You can literally start with this:
test_qa = [
    {
        'query': 'What is RAG?',
        'document': 'RAG stands for Retrieval Augmented Generation. It is a technique that combines information retrieval with generation.',
        'answer': 'A technique combining retrieval with generation'
    }
]

# And this is enough to test your RAG pipeline
# Then download real data when ready

# Your RAG code will work the same way!
```

You can literally start building RAG TODAY with one QA pair if you want.

But get the real data (Path A: 15 min) and you'll have a proper test set. 🎯
