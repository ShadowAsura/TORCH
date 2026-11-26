"""
TORCH core: sans-I/O state machines for decentralized GPU scheduling.

This package must not import asyncio, time, socket, httpx, fastapi or simpy.
Runtimes (simulator, HTTP) live outside the core and drive it.
"""
