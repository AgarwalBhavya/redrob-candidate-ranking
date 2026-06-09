# Hackathon Presentation Deck — Intelligent Candidate Discovery & Ranking

*This document serves as the slide-by-slide structure for your presentation deck. Copy this content into your slides (PowerPoint/Google Slides) and export it as a PDF to submit.*

---

## Slide 1: Title Slide
* **Title**: AI Recruiter: Intelligent Candidate discovery & ranking system
* **Subtitle**: A Hybrid Two-Stage Retrieval and Re-ranking System for Senior Machine Learning Engineers (applied NLP/IR)
* **Team**: Team Antigravity
* **Primary Contact**: Bhavya Agarwal
* **GitHub**: https://github.com/AgarwalBhavya/redrob-candidate-ranking

---

## Slide 2: The Core Problem & Constraints
### The Challenge
* Simple keyword matching fails to distinguish actual role fit from keyword stuffers.
* Recruiters miss out on "shippers" (practical product engineers) vs. academic-only researchers.
* Availability factors (unresponsiveness, high notice periods, platform inactivity) are often ignored by static keyword matchers.

### The Technical Constraints
* **Compute Limit**: CPU-only execution (no GPU), ≤ 16 GB RAM.
* **Network Limit**: Off-network ranking (no OpenAI, Anthropic, or external API calls).
* **Latency Limit**: Under 5 minutes wall-clock time for 100,000 candidates.

---

## Slide 3: High-Level System Architecture
*We use a two-stage information retrieval (IR) pipeline to achieve high semantic accuracy and lightning-fast speed.*

```
Candidates (100k) 
      │
      ▼
┌─────────────────────────────────┐
│ Dynamic Honeypot Filter         │  <── Filters out 181 logical anomaly profiles
└─────────────────────────────────┘
      │ (Valid: 99,819)
      ▼
┌─────────────────────────────────┐
│ Stage 1: TF-IDF Keyword Match   │  <── Fast pre-filtering (Narrows 99k to 2k in <3 sec)
└─────────────────────────────────┘
      │ (Shortlist: 2,000)
      ▼
┌─────────────────────────────────┐
│ Stage 2: Semantic Similarity    │  <── SentenceTransformers (all-MiniLM-L6-v2) offline
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Heuristic Multiplier Scoring    │  <── YoE, Title, Pedigree, Location, Notice Period, Activity
└─────────────────────────────────┘
      │
      ▼
Top 100 Shortlist (CSV)
```

---

## Slide 4: Anomaly Detection: Filtering the 181 Honeypots
*To protect the ranking pipeline, our system dynamically detects and discards profiles with logical contradictions (relevance tier 0).*

### Detection Logic
1. **Experience Inflation**: Individual job duration is longer than the candidate's total years of experience.
2. **Date range mismatch**: The difference between the date range (start/end) and reported `duration_months` is > 12 months.
3. **Keyword Stuffing**: Candidate lists expert/advanced proficiency in multiple skills but claims `0` months of actual experience.
4. **Chronological anomalies**: Earliest career start date is > 6 years before university education start year.
5. **Signal anomalies**: Range errors in Redrob platform scores (e.g. completeness score > 100).

*Result: Successfully identified and excluded exactly 181 honeypot candidates.*

---

## Slide 5: Stage 2: Multi-Dimensional Heuristic Scoring
*Once the semantic similarity is calculated, we apply heuristic multipliers based on specific filters described in the Job Description:*

1. **Years of Experience (YoE) Score**: Target: 5-9 years. Sweet spot: 6-8 years (multiplier `1.0`). Penalizes profiles below 4 or above 11 years.
2. **Role Fit Score**: Analyzes headlines and current/past titles for MLE, NLP, and Search terms. Downweights adjacent non-technical roles (e.g., Marketing, HR, Operations Managers).
3. **Pedigree Score (Consulting Company Penalty)**: Downweights candidates whose entire career history is spent at service giants (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) without product company experience (multiplier `0.3`).
4. **Location & Notice Period Match**: Shorter notice periods (< 30 days) and candidates residing in India (Pune, Noida, Bangalore, Delhi NCR, Hyderabad) receive maximum weights.
5. **Activity Factor**: Penalizes candidates with low recruiter response rates and long inactivity (last login > 6 months).

---

## Slide 6: Model Explainability & Custom Reasoning
*Our system generates natural, factually accurate justifications tailored for every candidate in the shortlist, avoiding repetitive templates or hallucinations:*

* **Rank 1 Candidate (CAND_0011687)**:
  > "Top-tier candidate with 7.8 years applied NLP/IR experience; currently Senior NLP Engineer at Niramai in Indore, Madhya Pradesh. Demonstrated expertise in FAISS, Embeddings and product engineering background with a short 15-day notice period."
* **Rank 3 Candidate (CAND_0000031)**:
  > "Exceptional ML engineer with 6.0 years of experience and deep expertise in FAISS, Pinecone, Machine Learning; currently Recommendation Systems Engineer at Swiggy. Deployed embedding-based search at Swiggy, showing strong alignment with the JD, alongside active platform engagement (91% response rate)."

---

## Slide 7: Verification & Results
### Auto-Validator Compliance
* Ran `validate_submission.py` successfully: **"Submission is valid."**
* Confirms UTF-8 encoding, exact 100-candidate size, monotonic non-increasing scores, and ascending candidate_id tie-breaking.

### Performance Summary
* **Execution Time**: **< 15 seconds** for 100k candidates on CPU.
* **Memory Usage**: **< 1.2 GB** RAM.
* **Honeypot Rate**: **0%** in the top 100 shortlist (disqualification rate is > 10%).
* **Quality**: Surfaced highly relevant Senior Machine Learning Engineers with proven NLP/IR search/retrieval backgrounds from India's top product engineering spaces (CRED, Swiggy, Rephrase.ai, Paytm, Sarvam AI, Ola, etc.).
