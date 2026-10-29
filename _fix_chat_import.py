with open(r"C:\Users\Himanshu\Downloads\Document-Ai-branch-vm2\backend-v2\trial-workflow\src\examples\ecommerce_claims\activities.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Replace the import line
content = content.replace(
    "from mistralai.client.chat import Chat",
    "from mistralai.client import Mistral",
    1
)

# Fix 2: Replace Chat(...) in _get_mistral_client helper (2 occurrences)
content = content.replace(
    "return Chat(api_key=api_key, server_url=server_url)",
    "return Mistral(api_key=api_key, server_url=server_url)",
    1
)
content = content.replace(
    "return Chat(api_key=api_key)",
    "return Mistral(api_key=api_key)",
    1
)

# Fix 3: Replace all inline Chat(api_key=...) instantiations in activity functions
content = content.replace(
    'Chat(api_key=os.getenv("MISTRAL_API_KEY", ""), server_url=os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL"))',
    'Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""), server_url=os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL"))'
)

with open(r"C:\Users\Himanshu\Downloads\Document-Ai-branch-vm2\backend-v2\trial-workflow\src\examples\ecommerce_claims\activities.py", "w", encoding="utf-8") as f:
    f.write(content)

# Verify the changes
import re
remaining_chat = re.findall(r"\bChat\(", content)
remaining_import = "from mistralai.client.chat import Chat" in content
print(f"Remaining Chat( calls: {len(remaining_chat)}")
print(f"Old import still present: {remaining_import}")
print(f"Mistral import present: {'from mistralai.client import Mistral' in content}")

# Count Mistral( replacements
mistral_inits = re.findall(r"\bMistral\(api_key=", content)
print(f"Mistral(api_key=...) occurrences: {len(mistral_inits)}")
