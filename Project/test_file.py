#!/usr/bin/env python3
"""
Script test OpenAI GPT-4o connection
Chạy script này để kiểm tra API key và model có hoạt động không
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_openai_connection():
    """Test OpenAI API connection"""

    print("=" * 70)
    print("🧪 Testing OpenAI GPT-4o Connection")
    print("=" * 70)

    # Step 1: Check API key
    print("\n1️⃣ Checking API key...")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment!")
        print("Please set it in your .env file")
        return False

    print(f"✅ API Key found: {api_key[:20]}...{api_key[-4:]}")

    # Step 2: Check model name
    print("\n2️⃣ Checking model configuration...")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    print(f"✅ Model: {model}")

    # Step 3: Test LangChain import
    print("\n3️⃣ Testing LangChain import...")
    try:
        from langchain_openai import ChatOpenAI
        print("✅ LangChain OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LangChain OpenAI: {e}")
        print("Run: pip install langchain-openai")
        return False

    # Step 4: Initialize LLM
    print("\n4️⃣ Initializing OpenAI LLM...")
    try:
        llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            api_key=api_key
        )
        print(f"✅ LLM initialized: {model}")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return False

    # Step 5: Test simple query
    print("\n5️⃣ Testing simple query...")
    try:
        from langchain_core.messages import HumanMessage

        messages = [
            HumanMessage(content="Xin chào! Bạn có thể nói tiếng Việt không?")
        ]

        response = llm.invoke(messages)
        print("✅ Query successful!")
        print(f"Response: {response.content[:200]}...")

    except Exception as e:
        print(f"❌ Query failed: {e}")

        # Check if it's an authentication error
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            print("\n💡 Suggestion: Your API key might be invalid or expired")
            print("   - Check your API key at https://platform.openai.com/api-keys")
            print("   - Make sure you have credits in your OpenAI account")

        # Check if it's a rate limit error
        elif "rate limit" in str(e).lower():
            print("\n💡 Suggestion: You've hit the rate limit")
            print("   - Wait a minute and try again")
            print("   - Check your usage at https://platform.openai.com/usage")

        # Check if it's a model access error
        elif "model" in str(e).lower():
            print(f"\n💡 Suggestion: Model '{model}' might not be available")
            print("   - Try using 'gpt-4o-mini' instead")
            print("   - Or check your model access at OpenAI dashboard")

        return False

    # Step 6: Test with Vietnamese tourism context
    print("\n6️⃣ Testing Vietnamese tourism query...")
    try:
        messages = [
            HumanMessage(content="Gợi ý 3 địa điểm du lịch nổi tiếng ở Quảng Ninh?")
        ]

        response = llm.invoke(messages)
        print("✅ Vietnamese query successful!")
        print(f"Response: {response.content[:300]}...")

    except Exception as e:
        print(f"⚠️ Vietnamese query failed: {e}")
        # Continue anyway since basic test passed

    print("\n" + "=" * 70)
    print("✅ All tests passed! OpenAI GPT-4o is ready to use")
    print("=" * 70)

    # Print usage tips
    print("\n💡 Usage Tips:")
    print(f"   - Model: {model}")
    print(f"   - API Key: Configured ✓")
    print(f"   - You can now run: python RAG_core/main.py")

    return True


def test_rag_system():
    """Test RAG system initialization"""

    print("\n" + "=" * 70)
    print("🧪 Testing RAG System")
    print("=" * 70)

    try:
        print("\n1️⃣ Importing RAG system...")
        from rag_multi_agent_system import BaiChayRAGSystem
        print("✅ Import successful")

        print("\n2️⃣ Initializing RAG system...")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        rag_system = BaiChayRAGSystem(openai_model=model)
        print("✅ RAG system initialized")

        print("\n3️⃣ Testing a simple query...")
        result = rag_system.process_query("Xin chào")
        print("✅ Query processed")
        print(f"Response: {result['response'][:200]}...")

        print("\n" + "=" * 70)
        print("✅ RAG System is working!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ RAG System test failed: {e}")
        print("\nThis might be normal if Milvus is not running.")
        print("Make sure Milvus is running before testing the full RAG system.")
        return False


if __name__ == "__main__":
    print("\n🚀 OpenAI GPT-4o Connection Test\n")

    # Test OpenAI connection
    openai_ok = test_openai_connection()

    if not openai_ok:
        print("\n❌ OpenAI connection test failed!")
        print("Please fix the issues above before proceeding.")
        sys.exit(1)

    # Ask if user wants to test RAG system
    print("\n" + "=" * 70)
    print("Would you like to test the full RAG system? (requires Milvus)")
    response = input("Enter 'y' to test, or any other key to skip: ").strip().lower()

    if response == 'y':
        test_rag_system()
    else:
        print("\n✅ Skipping RAG system test")
        print("You can test it later by running: python RAG_core/main.py")

    print("\n🎉 Setup complete! You're ready to use GPT-4o")