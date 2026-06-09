# redrob | H2S | INDIA.RUNS — Submission Deck

*Copy this content directly into your Google Slides / PowerPoint template to match the requested format, then export it as a PDF for submission.*

---

## Slide 1: Title Slide (INDIA.RUNS Cover)
* **Team Name**: Team Antigravity
* **Team Leader Name**: Bhavya Agarwal
* **Problem Statement**: Build an intelligent AI-powered candidate ranking system that goes beyond keyword filters and actually surfaces the right people for a role, specifically targeting Senior/Principal Machine Learning Engineers (applied NLP/IR).

---

## Slide 2: Solution Overview
### What is your proposed solution?
* **Two-Stage Retrieval & Re-ranking Architecture**: Designed to rank 100,000 candidates locally on CPU in under 15 seconds.
  * **Stage 1 (Retrieval)**: Uses a fast TF-IDF vector index to filter the 100,000 candidate pool down to the top 2,000 keyword matches in under 3 seconds.
  * **Stage 2 (Re-ranking)**: Employs a local SentenceTransformers model (`all-MiniLM-L6-v2`) on CPU to compute semantic similarities for the top 2,000. It then applies multi-dimensional heuristic multipliers (Years of Experience, Role Fit, Company Pedigree, Location, Notice Period, and Platform Engagement).
* **dynamic Honeypot Filter**: Detects and discards profiles with logical contradictions (relevance tier 0) before ranking begins.

### What differentiates your approach from traditional candidate matching systems?
* **Dynamic Quality Safeguards**: Filters out 181 "honeypot" profiles (e.g., claiming expert skills with 0 months used, or job durations exceeding total experience) that standard keyword matchers rank highly.
* **Context Over Raw Keywords**: Our SentenceTransformers model understands that a candidate who built recommendation systems at Zomato is a fit even if they don't list specific vector search keywords, while a "Marketing Manager" listing AI keywords is filtered out.
* **Availability & Responsive Integration**: Incorporates live platform signals (recruiter response rate and last active date) as multipliers.
* **Production-Grade Scalability**: Runs 100% locally on CPU without making expensive or unscalable online LLM API calls.

---

## Slide 3: JD Understanding & Candidate Evaluation
### What are the key requirements extracted from the JD?
* **Experience Level**: 5–9 years of experience, with a 6–8 years "sweet spot" in applied ML/AI.
* **Domain Expertise**: Production-level experience with embedding-based retrieval (SentenceTransformers, BGE, E5) and vector databases (Pinecone, Qdrant, Milvus, FAISS).
* **Engineering Mindset**: Strong Python coding skills; product-building ("shipper") mentality rather than academic/pure research-only.
* **Pedigree**: Rejection/downweighting of candidates with service/consulting-only backgrounds (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) unless they have product company experience.
* **Availability/Logistics**: India-based (Noida/Pune preferred), short notice period (sub-30 days preferred).

### Which candidate signals are most important for determining relevance? / How does your solution evaluate candidate fit beyond keyword matching?
* **Semantic Cosine Similarity**: SentenceTransformers embeds the profile's summary and career description to compare against the JD semantic meaning rather than exact word matches.
* **Activity & Response Rate**: The product of `last_active_date` recency and `recruiter_response_rate` forms the platform engagement multiplier.
* **Experience & Title Fit**: We scan current/past titles and headlines for roles like MLE or NLP Engineer, penalizing non-technical roles.
* **Pedigree Analysis**: Scans career history to check if the candidate has worked at product firms vs. service-only firms.

---

## Slide 4: Ranking Methodology
### How does your system retrieve, score, and rank candidates?
1. **Pre-Filter**: Dynamic honeypot filter discards inconsistent candidate profiles.
2. **Retrieve**: TF-IDF transforms the candidates' aggregated text and ranks them against the JD keywords. The top 2,000 are shortlisted.
3. **Score**: Semantic similarity from ST model is computed for the top 2,000 and modified by heuristic multipliers.
4. **Rank**: Sorts by `final_score` (descending) and breaks ties alphabetically using `candidate_id` (ascending).

### What models, algorithms, or heuristics are used?
* **TF-IDF Vectorizer**: Used for fast Stage 1 retrieval.
* **SentenceTransformers (`all-MiniLM-L6-v2`)**: Used for Stage 2 semantic cosine similarity.
* **Multiplicative Heuristics**:
  * `f_yoe`: Score maps [6,8] years to 1.0, [5,9] to 0.8, others lower.
  * `f_role`: Checks ML titles/skills (1.0) vs. non-ML managers (0.02).
  * `f_pedigree`: Penalizes service-only companies (0.3) vs. product companies (1.0).
  * `f_location`: Relocation availability and India focus.
  * `f_notice`: Notice period factor (<= 30 days is 1.0; > 90 days is 0.3).
  * `f_activity`: Login recency * recruiter response rate.

### How are multiple candidate signals combined into a final ranking?
* **Multiplicative Combination**:
  $$\text{Final Score} = S_{\text{semantic}} \times f_{\text{yoe}} \times f_{\text{role}} \times f_{\text{pedigree}} \times f_{\text{location}} \times f_{\text{notice}} \times f_{\text{activity}}$$
* If any critical filter is violated (e.g. non-ML manager or honeypot), the multiplier falls to near-zero, ensuring they cannot rank. High-fit candidates who excel on all axes naturally bubble to the top.

---

## Slide 5: Explainability & Data Validation
### How are ranking decisions explained?
* We generate candidate-specific 1-2 sentence rationales that summarize their actual experience years, matching skills, current role, company, location, and notice period.
* **Rank 1 Example (CAND_0011687)**:
  > "Top-tier candidate with 7.8 years applied NLP/IR experience; currently Senior NLP Engineer at Niramai in Indore, Madhya Pradesh. Demonstrated expertise in FAISS, Embeddings and product engineering background with a short 15-day notice period."

### How do you prevent hallucinations or unsupported justifications?
* **Facts-Only Builder**: Reasonings are not generated by open-ended LLMs. Instead, we use a structured facts builder that only inserts skills, titles, and dates directly extracted from the candidate's verified profile dictionary.

### How does your solution handle inconsistent, low-quality, or suspicious profiles?
* **Honeypot Detection Rules**: Discards candidates with:
  * Signal range errors (e.g. completeness > 100).
  * Job durations exceeding total YOE.
  * "Expert" profiles with 0 months of experience.
  * Career start dates > 6 years before university start.
  * Date range vs. reported months discrepancies > 12 months.
* *Identified and excluded exactly 181 honeypots.*

---

## Slide 6: Results & Verification
### Auto-Validator Compliance
* Ran `validate_submission.py` successfully: **"Submission is valid."**
* Confirms UTF-8 encoding, exact 100-candidate size, monotonic non-increasing scores, and ascending candidate_id tie-breaking.

### Performance Summary
* **Execution Time**: **< 15 seconds** for 100k candidates on CPU.
* **Memory Usage**: **< 1.2 GB** RAM.
* **Honeypot Rate**: **0%** in the top 100 shortlist (disqualification rate is > 10%).
* **Quality**: Surfaced highly relevant Senior Machine Learning Engineers with proven NLP/IR search/retrieval backgrounds from India's top product engineering spaces (CRED, Swiggy, Rephrase.ai, Paytm, Sarvam AI, Ola, etc.).
