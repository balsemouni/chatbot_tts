import asyncio
from agent.voice_agent import UltraLowLatencyVoiceAgent

if __name__ == "__main__":
    agent = UltraLowLatencyVoiceAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\n⏹️ Session closed")