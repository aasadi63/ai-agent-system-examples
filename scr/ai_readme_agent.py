#!/usr/bin/env python3
"""
GitHub README Table Summarization Agent
Uses LangChain and OpenAI to extract and summarize tables from GitHub repository READMEs
"""

import os
import re
from io import StringIO
from dotenv import load_dotenv
import pandas as pd
from github import Github
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
REPO_NAME = os.getenv("REPO_NAME", "aasadi63/ai-agent-system-examples")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def extract_tables_from_markdown(markdown_text):
    """Extract tables from markdown text and return as list of DataFrames."""
    tables = []
    pattern = r"\|(.+?)\|\n\|(?:[-:| ]+)\|\n((?:\|.*\|\n?)+)"
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    for header, body in matches:
        full_table = f"|{header}|\n|---|\n{body.strip()}"
        try:
            df = pd.read_csv(
                StringIO(full_table), 
                sep="|", 
                engine='python', 
                skipinitialspace=True
            )
            # Drop extra unnamed index columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            if not df.empty:
                tables.append(df)
        except Exception as e:
            print(f"  ⚠ Warning: Could not parse a table: {e}")

    return tables


def get_github_readme(repo_name):
    """Fetch README content from GitHub repository."""
    try:
        print(f"📡 Fetching README from {repo_name}...")
        g = Github()  # Anonymous access for public repos
        repo = g.get_repo(repo_name)
        readme = repo.get_readme()
        content = readme.decoded_content.decode("utf-8")
        print(f"✓ Successfully fetched README ({len(content)} characters)")
        return content
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def summarize_with_llm(tables_markdown):
    """Use OpenAI to generate summaries of the tables."""
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key not configured in .env file"
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert data analyst specializing in understanding structured data.

Your task is to analyze markdown tables and provide clear, concise summaries.

For each table, explain:
- What type of information it contains
- The purpose or use case
- Key patterns or notable aspects

Be specific and informative but concise."""),
        ("human", """Here are tables extracted from a GitHub repository README:

{tables_info}

Please analyze each table and provide a comprehensive summary.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    print("🧠 Sending tables to AI for analysis...")
    
    try:
        summary = chain.invoke({"tables_info": tables_markdown})
        return summary
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("🤖 GitHub README Table Summarization Agent")
    print(f"📊 Repository: {REPO_NAME}")
    print("="*80 + "\n")
    
    # Step 1: Fetch README from GitHub
    readme_content = get_github_readme(REPO_NAME)
    if not readme_content:
        print("❌ Failed to fetch README. Exiting.")
        return
    
    # Step 2: Extract tables
    print("\n📋 Extracting tables from README...")
    tables = extract_tables_from_markdown(readme_content)
    
    if not tables:
        print("❌ No tables found in the README.")
        return
    
    print(f"✓ Found {len(tables)} table(s)\n")
    print("="*80)
    
    # Display extracted tables
    tables_markdown = ""
    for i, df in enumerate(tables, 1):
        print(f"\n### Table {i}:")
        table_md = df.to_markdown(index=False)
        print(table_md)
        print("-"*80)
        tables_markdown += f"\n### Table {i}:\n{table_md}\n\n"
    
    # Step 3: Summarize with LLM
    print("\n" + "="*80)
    print("🤖 AI Analysis")
    print("="*80 + "\n")
    
    summary = summarize_with_llm(tables_markdown)
    
    print("\n" + "="*80)
    print("📝 Summary:")
    print("="*80)
    print(summary)
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
