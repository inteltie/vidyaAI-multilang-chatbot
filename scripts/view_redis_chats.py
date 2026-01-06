#!/usr/bin/env python3
"""Simple script to view conversation history stored in Redis."""

import asyncio
import json
import sys
import os
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redis.asyncio as aioredis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def view_all_sessions():
    """List all chat sessions in Redis."""
    client = aioredis.from_url("redis://localhost:6379/0")
    
    try:
        # Get all keys matching chat:*
        keys = await client.keys("chat:*")
        
        if not keys:
            print("No chat sessions found in Redis.")
            return
        
        print(f"\nFound {len(keys)} chat session(s):\n")
        print("=" * 80)
        
        for key in keys:
            session_id = key.decode('utf-8').replace('chat:', '')
            print(f"\n📝 Session ID: {session_id}")
            print("-" * 80)
            
            # Get all messages for this session
            messages = await client.lrange(key, 0, -1)
            
            if not messages:
                print("  (empty)")
                continue
            
            for i, msg in enumerate(messages, 1):
                try:
                    data = json.loads(msg)
                    role = data.get('role', 'unknown')
                    content = data.get('content', '')
                    
                    emoji = "👤" if role == "user" else "🤖"
                    print(f"\n  {emoji} {role.upper()}:")
                    print(f"     {content}")
                except json.JSONDecodeError:
                    print(f"  {i}. (invalid JSON: {msg})")
            
            print()
        
        print("=" * 80)
        
    finally:
        await client.close()


async def view_specific_session(session_id: str):
    """View a specific session's conversation history."""
    client = aioredis.from_url("redis://localhost:6379/0")
    
    try:
        key = f"chat:{session_id}"
        messages = await client.lrange(key, 0, -1)
        
        if not messages:
            print(f"No messages found for session: {session_id}")
            return
        
        print(f"\n📝 Session: {session_id}")
        print("=" * 80)
        
        for msg in messages:
            try:
                data = json.loads(msg)
                role = data.get('role', 'unknown')
                content = data.get('content', '')
                
                emoji = "👤" if role == "user" else "🤖"
                print(f"\n{emoji} {role.upper()}:")
                print(f"   {content}")
            except json.JSONDecodeError:
                print(f"(invalid JSON: {msg})")
        
        print("\n" + "=" * 80)
        
    finally:
        await client.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # View specific session
        session_id = sys.argv[1]
        asyncio.run(view_specific_session(session_id))
    else:
        # View all sessions
        asyncio.run(view_all_sessions())
