from dotenv import load_dotenv
from github import Github, Auth
import os

load_dotenv()
token = os.getenv('GITHUB_TOKEN')
repo_name = os.getenv('REPO_NAME')

print('Testing GitHub connection...')
auth = Auth.Token(token)
g = Github(auth=auth)
repo = g.get_repo(repo_name)
print(f'Successfully connected to: {repo.full_name}')
readme = repo.get_readme()
print(f'README size: {len(readme.decoded_content)} bytes')
print('All tests passed')
