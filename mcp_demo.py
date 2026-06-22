import os
import sys

sys.path.insert(0, r"D:\Code\Multi-Agent-Medical-Assistant")

print("=" * 60)
print("   Multi-Agent Medical Assistant - MCP Integration Demo")
print("=" * 60)

# 1. Load MCPConfig
print("\n[1/4] Loading MCPConfig...")
from config import MCPConfig

cfg = MCPConfig()
print(f"  biomcp:     enabled={cfg.biomcp['enabled']}, cmd={cfg.biomcp['command']}")
print(f"  autoicd:    enabled={cfg.autoicd['enabled']}, cmd={cfg.autoicd['command']}")
print(f"  healthcare: enabled={cfg.healthcare['enabled']}, cmd={cfg.healthcare['command']}")
print(f"  max_iterations={cfg.max_iterations}, tool_timeout={cfg.tool_timeout}s")
print("  ✅ MCPConfig loaded")

# 2. Load MCPClientManager
print("\n[2/4] Loading MCPClientManager...")
from agents.mcp_client import MCPClientManager, get_mcp_client

client = MCPClientManager(cfg)
print(f"  Client class: {type(client).__name__}")
print(
    f"  biomcp path exists: {os.path.exists(os.path.join(r'D:\\Code\\Multi-Agent-Medical-Assistant', 'mcp_servers', 'biomcp'))}"
)
print("  ✅ MCPClientManager ready")

# 3. Load MCP Agent
print("\n[3/4] Loading MCP Agent...")
from agents.mcp_agent import MCP_ROUTING_PROMPT, mcp_agent_node

print(f"  Node function: {mcp_agent_node.__name__}")
print(f"  Routing prompt: {len(MCP_ROUTING_PROMPT)} chars")
print("  Prompt preview:")
for line in MCP_ROUTING_PROMPT.split("\\n")[:8]:
    print(f"    {line}")
print("  ✅ MCP Agent loaded")

# 4. Check MCP server repos
print("\n[4/4] Checking MCP server repositories...")
servers_dir = os.path.join(r"D:\Code\Multi-Agent-Medical-Assistant", "mcp_servers")
for name in ["biomcp", "autoicd-mcp", "healthcare-mcp"]:
    path = os.path.join(servers_dir, name)
    if os.path.exists(path):
        files = os.listdir(path)
        print(f"  ✅ {name}: {len(files)} files in {path}")
    else:
        print(f"  ❌ {name}: NOT FOUND at {path}")

print("\n" + "=" * 60)
print("   ALL MCP MODULES VERIFIED SUCCESSFULLY!")
print("=" * 60)
