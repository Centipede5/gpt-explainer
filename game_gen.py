import requests
import json

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}

# Original sentence to generate prompts from
sentence = "Bats used in professional baseball games must meet specific standards for size and weight".split(" ")

prediction = ""
# Loop through each word in the sentence and generate a prompt for it
for i in range(1, len(sentence)):
    prompt = " ".join(sentence[:i])
    
    data = {
        "model": "llama3.2",
        "prompt": prompt,
        "raw": True,
        "options": {
            #"temperature": 0,
            "num_predict": 1
        },
        "stream": False
    }
    
    # Send the POST request to generate the next word
    response = requests.post('http://localhost:11434/api/generate', headers=headers, data=json.dumps(data))
    
    # Extract and print the response as a continuous sentence without spaces
    full_response = response.json()['response'].replace(" ", "")
    prediction += full_response + " "
    print(f"Step {i}: {full_response}")

print("total prediction:", prediction)