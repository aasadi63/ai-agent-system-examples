
import os
from io import StringIO  # Add at top of your script
from dotenv import load_dotenv
import pandas as pd
from github import Github
from markdownify import markdownify as md
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage

#Load API keys from .env file
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME")
# Initialize GitHub client
g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)
readme = repo.get_readme()
readme_content = readme.decoded_content.decode("utf-8")

# 2. Extract tables from Markdown using Pandas
def extract_tables_from_markdown(markdown_text):
    import re
    tables = []
    pattern = r"\|(.+?)\|\n\|(?:[-:| ]+)\|\n((?:\|.*\|\n?)+)"
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    for header, body in matches:
        full_table = f"|{header}|\n|---|\n{body.strip()}"
        try:
            df = pd.read_csv(StringIO(full_table), sep="|", engine='python', skipinitialspace=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # drop extra index cols
            tables.append(df)
        except Exception as e:
            print("Error parsing table:", e)

    return tables


tables = extract_tables_from_markdown(readme_content)

# 3. Use HuggingFace model for summarization
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
    model_kwargs={"temperature": 0.5, "max_new_tokens": 300}
)

template = """
You are an assistant that summarizes model contract markdown tables.

Here is a table in markdown format:
{table_md}

Write a concise summary of what this table describes.
"""
prompt = PromptTemplate(input_variables=["table_md"], template=template)
chain = LLMChain(prompt=prompt, llm=llm)

# 4. Generate summaries for each table
for i, df in enumerate(tables):
    table_md = df.to_markdown(index=False)
    summary = chain.run({"table_md": table_md})
    print('='*40)
    print(f"📋 Table {i+1} Summary:\n{summary}\n")
