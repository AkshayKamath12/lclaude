import ollama
import time

print("Sending request to qwen2.5:7b-instruct...")
start_time = time.time()

response = ollama.chat(
    model='qwen2.5:7b-instruct',
    messages=[{'role': 'user', 'content': 'Write a 3 line python script to reverse a string. No explanation, just code.'}]
)

end_time = time.time()

print("\n--- RESPONSE ---")
print(response['message']['content'])
print("----------------")
print(f"\nTime taken: {end_time - start_time:.2f} seconds")