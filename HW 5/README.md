# Unit 5 - Software Engineering Security Review Agent

Uses synthetic sample files only and a local Ollama model.

## Run
1. Install Python 3.10+ and Ollama.
2. In PowerShell:
   ```powershell
   ollama pull qwen2.5:3b
   pip install -r requirements.txt
   python security_agent.py --demo
   ```
3. For a live local-model review:
   ```powershell
   python security_agent.py
   ```
4. To approve the local mock ticket (still no external transmission):
   ```powershell
   python security_agent.py --approve
   ```

The deterministic demo is recommended for reproducing the evidence in the report.
`mock_ticket_outbox.json` and `agent_audit.log` are generated locally when run.
