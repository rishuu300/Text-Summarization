import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit UI Setup
st.set_page_config(page_title="LangChain: Summarize YT/Website", page_icon='🦜')
st.title("🦜 LangChain: Summarize YT or Website Content")
st.subheader("Enter a YouTube or Website URL")

# Sidebar for API Key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Main Input
generic_url = st.text_input("Paste the URL here", label_visibility="collapsed")

# Define LLM 
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key, temperature=0.5)

# Prompt Template
prompt_template = """
Provide a clear and concise summary (max 300 words) of the content below:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# On Button Click
if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please enter both the Groq API Key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL. Please enter a proper website or YouTube link.")
    else:
        try:
            with st.spinner("Loading content and summarizing..."):
                # Load data from YouTube or URL
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
                    )
                    
                docs = loader.load()

                # Split long documents
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = splitter.split_documents(docs)

                # Create summarization chain
                chain = load_summarize_chain(llm, chain_type="map_reduce", prompt=prompt)
                output = chain.invoke(split_docs)

                # Display summary
                st.subheader("Summary")
                st.text_area("Output", output['output_text'], height=300)

        except Exception as e:
            st.error("An error occurred:")
            st.exception(e)