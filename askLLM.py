# import asyncio
from chuk_llm import get_client

async def ask_llm(prompt):
    client = get_client(provider="ollama", model="granite3.3:latest")

    messages = [
        {"role": "system", "content": f"You are a study buddy chatbot to help students with study materials, and eductional needs. Answer the user's question {prompt}"},
        {"role": "user", "content": "f{prompt}"}
    ]

    complete = await client.create_completion(messages=messages)
    result = complete["response"]

    return result


# if __name__ == "__main__":
#     asyncio.run(main())