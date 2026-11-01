# Fictional application configuration for the Unit 5 assignment.
# The credential below is deliberately fake and cannot authenticate anywhere.
API_TOKEN = "sk-demo-FAKE-1234567890"

def calculate_total(items):
    # Example code for the agent to inspect.
    return sum(item["price"] for item in items)
