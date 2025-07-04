import torch
import whisper
import numpy as np
import sounddevice as sd
import tempfile
import wave
import os
import requests
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import re

# Load Whisper model (Tiny English for fast transcription)
model = whisper.load_model("tiny.en")

# Audio parameters
SAMPLE_RATE = 16000  # Whisper expects 16kHz input
DURATION = 10  # Duration for recording

print("Using device:", "GPU" if torch.cuda.is_available() else "CPU")

# Function to record audio
def record_audio(duration=DURATION, samplerate=SAMPLE_RATE):
    """Records audio for a fixed duration and returns numpy array"""
    print("Listening...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()  # Wait for recording to finish
    print("Recording complete!")
    return np.squeeze(audio_data)  # Convert from 2D to 1D array

# Function to save recorded audio as a WAV file
def save_wav(audio_data, filename):
    """Saves recorded audio as a WAV file"""
    audio_data = (audio_data * 32767).astype(np.int16)  # Convert to int16 format
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

# Function to detect multi-file requests
def is_multi_file_request(user_input):
    """Checks if the request involves creating/deleting/modifying multiple files"""
    patterns = [
        r'create\s+\d+\s+files',  # e.g., "create 10 files"
        r'named\s+\w+1\s+to\s+\d+',  # e.g., "named hi1 to 10"
        r'delete\s+files\s+\w+1\s+to\s+\w+\d+',  # e.g., "delete files new1.txt to new10.txt"
        r'modify\s+\d+\s+files',  # e.g., "modify 5 files"
    ]
    return any(re.search(pattern, user_input.lower()) for pattern in patterns)

# Function to get Linux command using Groq API with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.exceptions.ConnectionError)
)
def get_linux_command(user_input, is_multi_step, file_structure=None):
    """Sends user request to Llama 3 (8B) API and extracts only the Linux command"""
    # API_KEY = "KEY"  # Replace with your Groq API key
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # Base prompt for normal actions
    base_prompt = """
You are a Linux shell assistant. Your task is to return only the Linux command for the user's request.

- Do not explain anything.
- Do not include any additional text, context, or formatting outside the command.
- If the script involves writing a mail on the given topic or song or dialog or anything else do that completly into the file same with modification
- Always return a valid Linux command.
- If the request involves creating a single file, use `touch filename` for empty files or `echo` for files with content.
- If the request involves creating a directory, use `mkdir -p directoryname`.
- If the request involves writing text into a file, use `echo` (e.g., `echo "text" > filename`).
- If the request involves creating a C++ file, use `echo` to write valid C++ code into a `.cpp` file.
- If the request involves using `pip install`, prepend `sudo` for global installs: `sudo pip install package_name`.
- If a command involves installing software, updating packages, or modifying system settings, prepend `sudo`.
- If a command does not require `sudo`, generate it normally.
- Do NOT use bash expansions (e.g., `touch file{1..10}`) for multiple files; such requests should be handled as multi-step.
- Example for single file:
  User Request: Create a file named test.txt
  Response:
  ```bash
  touch test.txt
  ```
- Example for writing to a file:
  User Request: Write "Hello World" to test.txt
  Response:
  ```bash
  echo "Hello World" > test.txt
  ```
- Example for C++ file:
  User Request: create a cpp file to print random numbers btw 5 - 15
  Response:
  ```bash
  echo "#include <iostream>\n#include <cstdlib>\n#include <ctime>\nint main() {\n    std::srand(std::time(0));\n    int num = 5 + std::rand() % 11;\n    std::cout << num << std::endl;\n    return 0;\n}" > random.cpp
  ```
"""

    # Multi-step prompt for multiple file actions
    multi_step_prompt = """
You are a Linux shell assistant. Your task is to return only the Linux command for the user's request.

- Do not explain anything.
- Do not include any additional text, context, or formatting outside the command.
- If the script involves writing a mail on the given topic or song or dialog or anything else do that completly into the file same with modification
- Always return a valid Linux command.
- If the request involves creating, deleting, or modifying multiple files (e.g., 'create 10 files named hi1 to 10', 'delete files new1.txt to new10.txt'):
  - Generate a command that creates a Python script named `action_script.py` to perform the action.
  - Use `echo` to write the Python script content with proper indentation, no semicolons, and valid Python syntax.
  - The Python script must:
    - Perform the requested action (e.g., create files with `open(filename, 'w').close()`, delete with `os.remove(filename)`, or modify with file operations).
    - Handle errors (e.g., skip missing files with `try/except FileNotFoundError` for deletions).
    - Delete itself with `os.remove(__file__)` at the end.
  - Execute the script with `python3 action_script.py`.
  - Example for creating multiple files:
    User Request: create 10 files named hi1 to 10
    Response:
    ```bash
    echo "import os\nfiles = [f'hi{i}' for i in range(1, 11)]\nfor f in files:\n    open(f, 'w').close()\nos.remove(__file__)" > action_script.py; python3 action_script.py
    ```
  - Example for deleting multiple files:
    User Request: delete files new1.txt to new10.txt
    Response:
    ```bash
    echo "import os\nfiles = [f'new{i}.txt' for i in range(1, 11)]\nfor f in files:\n    try:\n        os.remove(f)\n    except FileNotFoundError:\n        pass\nos.remove(__file__)" > action_script.py; python3 action_script.py
    ```
  - Example for random file creation:
    User Request: create 2 files with random names
    Response:
    ```bash
    echo "import os\nimport random\nimport string\nfilenames = [''.join(random.choices(string.ascii_letters + string.digits, k=8)) + '.txt' for _ in range(2)]\nfor f in filenames:\n    open(f, 'w').close()\nos.remove(__file__)" > action_script.py; python3 action_script.py
    ```
- Avoid using bash loops (e.g., `touch file{1..10}`) or Python one-liners (e.g., `python3 -c "..."`).
- Ensure the Python script uses proper indentation and valid syntax.
"""

    # File structure prompt
    file_structure_prompt = """
You are a Linux shell assistant. Your task is to return only the Linux command for the user's request.

- Do not explain anything.
- Do not include any additional text, context, or formatting outside the command.
- Always return a valid Linux command.
- The user has specified a file structure (e.g., 'Folder1 > file1.txt' or 'Folder1 > Subfolder > file2.txt').
- Generate a command that creates a Python script named `action_script.py` to create the specified folder/file hierarchy.
- Use `echo` to write the Python script content with proper indentation and valid Python syntax.
- The Python script must:
  - Create directories with `os.makedirs(path, exist_ok=True)`.
  - Create files with `open(filename, 'w').close()`.
  - Handle the structure provided: {file_structure}.
  - Delete itself with `os.remove(__file__)` at the end.
- Execute the script with `python3 action_script.py`.
- Example:
  User File Structure: Folder1 > file1.txt
  Response:
  ```bash
  echo "import os\nos.makedirs('Folder1', exist_ok=True)\nopen('Folder1/file1.txt', 'w').close()\nos.remove(__file__)" > action_script.py; python3 action_script.py
  ```
- Example with nested structure:
  User File Structure: Folder1 > Subfolder > file2.txt
  Response:
  ```bash
  echo "import os\nos.makedirs('Folder1/Subfolder', exist_ok=True)\nopen('Folder1/Subfolder/file2.txt', 'w').close()\nos.remove(__file__)" > action_script.py; python3 action_script.py
  ```
- Ensure the Python script uses proper indentation and valid syntax.
"""

    # Select prompt based on options
    if file_structure:
        prompt = file_structure_prompt.format(file_structure=file_structure)
    else:
        prompt = multi_step_prompt if is_multi_step else base_prompt
    prompt += f"\nNow respond to this request:\nUser Request: {user_input}"

    data = {"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    response_json = response.json()

    # Debug: Print raw response
    print("\nRaw API Response:\n", json.dumps(response_json, indent=2))

    if "choices" in response_json and response_json["choices"]:
        command = response_json["choices"][0]["message"]["content"].strip()

        # Extract command if wrapped in triple backticks
        if "```bash" in command:
            command = command.split("```bash")[1].split("```")[0].strip()
        elif "```" in command:
            command = command.split("```")[1].split("```")[0].strip()

        return command
    else:
        print("Error: No valid command found in API response.")
        return ""

# Function to get user input (text or voice)
def get_user_input():
    """Prompts user to choose text or voice input and returns the input text"""
    print("\nChoose input method:")
    print("1. Text (type your request)")
    print("2. Voice (speak your request)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        user_text = input("\nEnter your request: ").strip()
        if user_text:
            print("\nYou entered:", user_text)
            return user_text
        else:
            print("\nError: Empty input.")
            return ""
    elif choice == "2":
        print(f"\nRecording for {DURATION} seconds...")
        audio_chunk = record_audio()
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            save_wav(audio_chunk, tmpfile.name)
            temp_path = tmpfile.name  # Store path for deletion

        # Transcribe using Whisper
        print("Transcribing...")
        result = model.transcribe(temp_path, fp16=False, language="en", without_timestamps=True, condition_on_previous_text=False)
        
        # Remove the temporary file
        os.remove(temp_path)

        # Extract transcribed text
        user_text = result["text"].strip()
        if user_text:
            print("\nYou said:", user_text)
            return user_text
        else:
            print("\nNo clear speech detected.")
            return ""
    else:
        print("\nInvalid choice. Please select 1 or 2.")
        return ""

# Function to choose execution option
def choose_execution_option(user_input):
    """Prompts user to choose between multi-step, normal, or file structure execution"""
    # Auto-detect multi-file requests
    auto_multi_step = is_multi_file_request(user_input)
    if auto_multi_step:
        print("\nDetected a multi-file request. Using Multi-step execution.")
        return "multi-step", None

    print("\nChoose execution option:")
    print("1. Multi-step (e.g., create/delete/modify multiple files with a script)")
    print("2. Normal (e.g., single file or simple command)")
    print("3. File structure (e.g., create Folder1 > file1.txt)")
    choice = input("Enter 1, 2, or 3: ").strip()
    if choice == "1":
        return "multi-step", None
    elif choice == "2":
        return "normal", None
    elif choice == "3":
        structure = input("\nEnter file structure (e.g., Folder1 > file1.txt): ").strip()
        if structure:
            print("\nFile structure:", structure)
            return "file-structure", structure
        else:
            print("\nError: Empty file structure. Defaulting to Normal.")
            return "normal", None
    else:
        print("\nInvalid choice. Defaulting to Normal.")
        return "normal", None

# Main execution
try:
    # Get user input (text or voice)
    user_text = get_user_input()

    if user_text:
        # Choose execution option
        execution_option, file_structure = choose_execution_option(user_text)
        is_multi_step = execution_option == "multi-step"
        is_file_structure = execution_option == "file-structure"

        # Get Linux command from Llama 3 via Groq API
        try:
            command = get_linux_command(user_text, is_multi_step, file_structure)
            if command:
                print("\nGenerated Command:\n", command)

                # Run the command
                print("\nExecuting command...\n")
                if command.startswith("cd "):
                    new_dir = command.split("cd ")[1].strip()
                    try:
                        os.chdir(new_dir)  # Change working directory
                        print(f"\nChanged directory to: {os.getcwd()}")
                    except FileNotFoundError:
                        print(f"\nError: Directory '{new_dir}' not found.")
                else:
                    os.system(command)
            else:
                print("\nNo valid command detected.")
        except requests.exceptions.ConnectionError as e:
            print(f"Network Error: Failed to connect to Groq API: {str(e)}")
            print("Please check your internet connection or try again later.")
    else:
        print("\nNo input provided.")

except KeyboardInterrupt:
    print("\nStopped.")
