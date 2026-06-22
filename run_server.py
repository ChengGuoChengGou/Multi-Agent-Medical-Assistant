#!/usr/bin/env python3
"""Simple launcher for uvicorn on port 8001."""
import uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, log_level="info")
