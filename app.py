import streamlit as st
from rag import retrieve, generate_answer


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Agriculture RAG Assistant",
    page_icon="🌱",
    layout="wide"
)


# ==========================================
# DOCUMENT NAME MAPPING
# ==========================================

DOCUMENT_NAMES = {
    "01_soil_health_management.pdf": "Soil Health Management",
    "02_npk_fertilizer_management.pdf": "NPK Fertilizer Management",
    "03_irrigation_water_management.pdf": "Irrigation & Water Management",
    "04_integrated_pest_management.pdf": "Integrated Pest Management",
    "05_crop_rotation_and_diversification.pdf": "Crop Rotation & Diversification",
    "06_precision_agriculture.pdf": "Precision Agriculture",
    "07_climate_smart_agriculture.pdf": "Climate-Smart Agriculture",
    "08_organic_and_sustainable_farming.pdf": "Organic & Sustainable Farming",
}


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

.main-title {
    font-size: 42px;
    font-weight: 750;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 17px;
    color: #8b8b8b;
    margin-bottom: 25px;
}

.info-box {
    padding: 15px 18px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 20px;
}

.source-box {
    padding: 14px 16px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.35);
    margin-bottom: 10px;
}

.source-title {
    font-size: 16px;
    font-weight: 600;
}

.source-page {
    font-size: 14px;
    color: #888;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🌱 Agriculture RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions grounded in a curated agriculture knowledge base.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.caption("8 agriculture documents indexed")

    for document in DOCUMENT_NAMES.values():
        st.write(f"📄 {document}")

    st.divider()

    st.subheader("⚙️ RAG Pipeline")

    st.markdown("""
    **1.** User question  
    ↓  
    **2.** Query embedding  
    ↓  
    **3.** FAISS semantic search  
    ↓  
    **4.** Top relevant chunks  
    ↓  
    **5.** OpenAI generation
    """)

    st.divider()

    st.caption(
        "FAISS • Sentence Transformers • OpenAI"
    )


# ==========================================
# ABOUT RAG
# ==========================================

with st.expander("🧠 How does this assistant work?"):

    st.markdown("""
    This application uses **Retrieval-Augmented Generation (RAG)**.

    Instead of asking the language model to answer directly:

    1. Your question is converted into an **embedding**.
    2. **FAISS** searches the agriculture vector database.
    3. The most relevant PDF chunks are retrieved.
    4. Those chunks are supplied to the **OpenAI model as context**.
    5. The model generates an answer grounded in the retrieved documents.

    You can inspect the retrieved context below every answer.
    """)


# ==========================================
# EXAMPLE QUESTIONS
# ==========================================

st.markdown("#### 💬 Try asking")

examples = [
    "What are the benefits of crop rotation?",
    "What are the roles of nitrogen, phosphorus and potassium?",
    "How can farmers improve water-use efficiency?",
    "How can fertilizer and irrigation management work together?"
]

cols = st.columns(2)

for i, example in enumerate(examples):
    cols[i % 2].caption(f"• {example}")


# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_area(
    "Ask an agriculture question",
    placeholder="Example: How can fertilizer and irrigation management work together?",
    height=110
)

ask_button = st.button(
    "🔍 Ask Agriculture Assistant",
    type="primary",
    use_container_width=True
)


# ==========================================
# PROCESS QUESTION
# ==========================================

if ask_button:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        # ----------------------------------
        # RETRIEVAL
        # ----------------------------------

        with st.spinner(
            "Searching the agriculture knowledge base..."
        ):

            retrieved_chunks = retrieve(
                question,
                top_k=3
            )


        # ----------------------------------
        # GENERATION
        # ----------------------------------

        with st.spinner(
            "Generating a grounded answer..."
        ):

            answer = generate_answer(
                question,
                retrieved_chunks
            )


        st.divider()


        # ==================================
        # ANSWER
        # ==================================

        st.subheader("💡 Answer")

        st.markdown(answer)


        # ==================================
        # SOURCES
        # ==================================

        st.subheader("📚 Sources")

        shown_sources = set()

        for chunk in retrieved_chunks:

            source_key = (
                chunk["source"],
                chunk["page"]
            )

            if source_key not in shown_sources:

                shown_sources.add(source_key)

                clean_name = DOCUMENT_NAMES.get(
                    chunk["source"],
                    chunk["source"]
                )

                st.markdown(
                    f"""
                    <div class="source-box">

                    <div class="source-title">
                    📄 {clean_name}
                    </div>

                    <div class="source-page">
                    Page {chunk["page"]}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ==================================
        # RETRIEVED CONTEXT
        # ==================================

        st.subheader("🔎 Retrieved Context")

        st.caption(
            "These are the document chunks retrieved by FAISS "
            "and supplied to the language model."
        )

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            clean_name = DOCUMENT_NAMES.get(
                chunk["source"],
                chunk["source"]
            )

            with st.expander(
                f"Chunk {i} — {clean_name} • Page {chunk['page']}"
            ):

                st.write(chunk["text"])

                st.caption(
                    f"FAISS distance: "
                    f"{chunk['distance']:.4f} "
                    "(lower = more similar)"
                )


        # ==================================
        # RAG TRACE
        # ==================================

        with st.expander("⚙️ View RAG Process"):

            st.markdown(
                f"""
                **User query**

                `{question}`

                ↓

                **Query converted into an embedding**

                ↓

                **FAISS searched 41 stored chunks**

                ↓

                **Top {len(retrieved_chunks)} chunks retrieved**

                ↓

                **Retrieved context + question sent to OpenAI**

                ↓

                **Grounded answer generated**
                """
            )