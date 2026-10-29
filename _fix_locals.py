with open(r"C:\Users\Himanshu\Downloads\Document-Ai-branch-vm2\backend-v2\trial-workflow\src\examples\ecommerce_claims\activities.py", "r", encoding="utf-8") as f:
    content = f.read()

old = '            span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": result.model_dump_json(), "arguments": locals(), "final_results": result.model_dump_json()}))'
new = '            span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": result.model_dump_json(), "final_results": result.model_dump_json()}))'

count = content.count(old)
print(f"Found {count} occurrences to replace")

content = content.replace(old, new)

with open(r"C:\Users\Himanshu\Downloads\Document-Ai-branch-vm2\backend-v2\trial-workflow\src\examples\ecommerce_claims\activities.py", "w", encoding="utf-8") as f:
    f.write(content)

remaining = content.count("locals()")
print(f"Remaining locals() calls: {remaining}")
print("Done.")
