import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# Import functions from our main rank.py script
from rank import (
    is_honeypot_candidate,
    build_search_text,
    build_semantic_text,
    calculate_heuristic_multipliers,
    generate_reasoning,
    REFERENCE_DATE
)

st.set_page_config(
    page_title="Team Antigravity - Recruiter AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Layout Setup
st.title("🤖 Recruiter AI: Candidate Discovery & Ranking")
st.markdown("### INDIA.RUNS Hackathon Submission — Team Antigravity")

# Sidebar information
st.sidebar.image("https://redrob.com/images/redrob-logo.png", width=150)
st.sidebar.header("System Statistics")
st.sidebar.info(
    "**Two-Stage Architecture**:\n"
    "- Stage 1: TF-IDF (Recall 100k -> 2k)\n"
    "- Stage 2: SentenceTransformers + Heuristics (Precision 2k -> 100)\n\n"
    "**Performance on 100k pool**:\n"
    "- **Runtime**: < 15 seconds\n"
    "- **Memory**: < 1.2 GB RAM\n"
    "- **Honeypot Filter**: 181/100,000"
)

st.sidebar.header("About the Team")
st.sidebar.markdown(
    "**Team Name**: Team Antigravity\n"
    "**Leader**: Bhavya Agarwal\n"
    "**Repo**: [GitHub Link](https://github.com/AgarwalBhavya/redrob-candidate-ranking)"
)

# Load local pre-computed submission.csv if present
csv_path = "submission.csv"
if os.path.exists(csv_path):
    df_shortlist = pd.read_csv(csv_path)
else:
    df_shortlist = None

# Tab organization
tab1, tab2, tab3 = st.tabs(["📋 Shortlist Explorer", "⚡ Custom Sandbox (Live Ranker)", "📊 Talent Pool Insights"])

with tab1:
    st.header("Top 100 Candidate Shortlist")
    st.markdown(
        "This list shows the best-fit candidates for the **Senior/Principal Machine Learning Engineer (applied NLP/IR)** role. "
        "Dynamic honeypot filters have been run, and scores combine semantic match with availability and company pedigree."
    )
    
    if df_shortlist is not None:
        # Search & Filter controls
        col1, col2 = st.columns([2, 1])
        with col1:
            search_query = st.text_input("🔍 Search candidates by ID or key reasoning keywords:", "")
        with col2:
            min_score = st.slider("Filter by minimum score:", 0.0, 1.0, 0.2, 0.05)
            
        filtered_df = df_shortlist[df_shortlist["score"] >= min_score]
        if search_query:
            filtered_df = filtered_df[
                filtered_df["candidate_id"].str.contains(search_query, case=False, na=False) |
                filtered_df["reasoning"].str.contains(search_query, case=False, na=False)
            ]
            
        st.dataframe(
            filtered_df,
            column_config={
                "candidate_id": "Candidate ID",
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "score": st.column_config.NumberColumn("Confidence Score", format="%.4f"),
                "reasoning": "Recruiter Decision & Reasoning"
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Shortlist CSV",
            data=csv_data,
            file_name="antigravity_shortlist.csv",
            mime="text/csv"
        )
    else:
        st.warning("Pre-computed `submission.csv` was not found. Please run the ranking engine first.")

with tab2:
    st.header("Candidate Discovery Sandbox")
    st.markdown(
        "Upload a small sample of candidates (JSON or JSONL, ≤ 100 profiles) to see the ranking model run **live** on CPU "
        "and produce a validated, scored shortlist."
    )
    
    uploaded_file = st.file_uploader("Choose a candidates file (.json or .jsonl)", type=["json", "jsonl"])
    
    if uploaded_file is not None:
        try:
            # Load the candidates
            candidates = []
            if uploaded_file.name.endswith(".jsonl"):
                for line in uploaded_file:
                    if line.strip():
                        candidates.append(json.loads(line.decode("utf-8")))
            else:
                # Normal JSON list
                candidates = json.loads(uploaded_file.read().decode("utf-8"))
                
            st.success(f"Successfully loaded {len(candidates)} candidates.")
            
            if st.button("🚀 Run Live Ranking Model"):
                with st.spinner("Executing Anomaly Checks, TF-IDF Retrieval, and SentenceTransformers semantic re-ranking..."):
                    # 1. Filter Honeypots
                    valid_candidates = [c for c in candidates if not is_honeypot_candidate(c)]
                    honeypot_diff = len(candidates) - len(valid_candidates)
                    
                    if not valid_candidates:
                        st.error("No valid candidates left after honeypot filtering!")
                    else:
                        st.info(f"Filtered out {honeypot_diff} honeypots. Proceeding with {len(valid_candidates)} candidates.")
                        
                        # 2. Stage 1 TF-IDF
                        search_texts = [build_search_text(c) for c in valid_candidates]
                        jd_query = "Machine Learning Engineer NLP Search Information Retrieval Ranking Embeddings Vector Database recommendation Python PyTorch"
                        
                        vectorizer = TfidfVectorizer(stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform(search_texts)
                        query_vector = vectorizer.transform([jd_query])
                        tfidf_scores = (tfidf_matrix * query_vector.T).toarray().flatten()
                        
                        # Sort and get top K
                        top_k = min(len(valid_candidates), 100)
                        top_k_indices = np.argsort(tfidf_scores)[::-1][:top_k]
                        stage1_candidates = [valid_candidates[idx] for idx in top_k_indices]
                        
                        # 3. Load model and score
                        # Try loading local model first, fall back to online
                        model_dir = "./model/all-MiniLM-L6-v2"
                        model_path = model_dir if os.path.exists(model_dir) else "all-MiniLM-L6-v2"
                        model = SentenceTransformer(model_path)
                        
                        semantic_query = (
                            "Senior Machine Learning Engineer NLP Search Information Retrieval. Deployed embeddings-based retrieval "
                            "systems (sentence-transformers, BGE, E5) and vector databases or hybrid search (Pinecone, Qdrant, Milvus, "
                            "FAISS, Elasticsearch) at scale. Strong Python coding. NLP and search ranking experience at product companies."
                        )
                        semantic_texts = [build_semantic_text(c) for c in stage1_candidates]
                        
                        embeddings = model.encode(semantic_texts, convert_to_tensor=True)
                        q_embedding = model.encode(semantic_query, convert_to_tensor=True)
                        cosine_scores = util.cos_sim(embeddings, q_embedding).cpu().numpy().flatten()
                        
                        # 4. Apply multipliers
                        scored_candidates = []
                        for i, cand in enumerate(stage1_candidates):
                            s_sem = max(0.0, float(cosine_scores[i]))
                            f_yoe, f_role, f_pedigree, f_location, f_notice, f_activity = calculate_heuristic_multipliers(cand)
                            final_score = s_sem * f_yoe * f_role * f_pedigree * f_location * f_notice * f_activity
                            
                            scored_candidates.append({
                                "candidate_id": cand["candidate_id"],
                                "score": final_score,
                                "candidate": cand
                            })
                            
                        # Sort
                        scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
                        
                        # Build Output DataFrame
                        output_rows = []
                        for rank_idx, item in enumerate(scored_candidates):
                            rank = rank_idx + 1
                            cid = item["candidate_id"]
                            score = item["score"]
                            reason = generate_reasoning(item["candidate"], rank)
                            output_rows.append({
                                "rank": rank,
                                "candidate_id": cid,
                                "score": score,
                                "reasoning": reason
                            })
                            
                        df_sandbox_out = pd.DataFrame(output_rows)
                        st.success("Ranking execution complete!")
                        
                        st.dataframe(
                            df_sandbox_out,
                            column_config={
                                "rank": "Rank",
                                "candidate_id": "Candidate ID",
                                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                                "reasoning": "Reasoning"
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Download sandbox output
                        sandbox_csv = df_sandbox_out.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Sandbox Shortlist CSV",
                            data=sandbox_csv,
                            file_name="sandbox_ranked_shortlist.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error parsing candidates: {e}")

with tab3:
    st.header("Talent Pool Demographics & Insights")
    st.markdown("Visual distribution of experience, location matches, and scores across the top 100 candidates.")
    
    if df_shortlist is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Shortlist Score Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df_shortlist["score"], bins=10, color="#6C5CE7", edgecolor="black", alpha=0.7)
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Count of Candidates")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Experience Level Insights (Top Candidates)")
            # Extract YoE from reasoning texts (safe regex parser)
            yoe_vals = []
            for r in df_shortlist["reasoning"]:
                m = re.search(r'(\d+\.\d+|\d+)\s+years', r)
                if m:
                    yoe_vals.append(float(m.group(1)))
            
            if yoe_vals:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(yoe_vals, bins=8, color="#00CEC9", edgecolor="black", alpha=0.7)
                ax.set_xlabel("Years of Experience")
                ax.set_ylabel("Number of Candidates")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
            else:
                st.info("Experience chart data could not be parsed.")
                
        # Additional metrics summary
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Candidates Evaluated", "100,000")
        with col_m2:
            st.metric("Honeypots Discarded", "181")
        with col_m3:
            st.metric("Best Shortlist Fit Score", f"{df_shortlist['score'].max():.4f}")
    else:
        st.warning("Shortlist stats not available. Run the ranking model first.")
