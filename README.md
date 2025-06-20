# 🔍 Search Engine Project

A simple **Search Engine** built in Python using Flask, NLTK, and TF‑IDF for indexing and retrieval.  
It processes XML documents (such as those from ClinicalTrials.gov) to enable quick, relevant searches.

---

## ⚙️ Features

- **Preprocesses XML documents**:
  - Extracts `<brief_title>` and `<brief_summary>`.
  - Cleans text, lowercases, removes stop words, and applies stemming.
- **Indexes documents using TF‑IDF**:
  - TF = Term Frequency
  - DF = Document Frequency
  - TF‑IDF = TF × log(N / DF)
- **Searches using Cosine Similarity**:
  - Returns results ranked by relevance.
- **Flask Web Interface**:
  - Simple and clean search page.
  - Displays Document ID, Score, Snippet, and Full View.
  - Shows Term‑level statistics (TF and TF‑IDF).
  - Displays Document statistics (length, term count).

---

## 🗂️ Directory Structure
Document_Search/
├─ templates/
├─ documents/
│ └─ sample_xml_files.xml
├─ app.py
├─ GroupName.txt
├─ Search_Engine_Documentation 
├─ vector_space_mode.py
