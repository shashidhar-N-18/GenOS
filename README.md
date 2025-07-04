# GenOS – Intelligent OS Voice Assistant

GenOS is an AI-powered voice assistant for Linux operating systems, enabling hands-free interaction through natural language commands. It leverages state-of-the-art speech-to-text and large language model (LLM) technologies to generate, interpret, and execute terminal commands in real-time.

---

## 🚀 **Features**

- 🎙️ **Voice Input Processing:** Uses **OpenAI Whisper** for accurate speech-to-text transcription.
- 🤖 **Natural Language Command Generation:** Employs **LLaMA 3** for command parsing, generation, and multi-step command chaining.
- 💻 **Shell Command Execution:** Safely executes commands on the Linux terminal with contextual awareness.
- 🔄 **Multi-step Command Handling:** Supports complex, sequential task execution based on user voice prompts.
- 🔒 **Secure Execution:** Incorporates checks to prevent harmful or unintended command execution.
- ⚡ **Hands-Free Operation:** Designed for enhanced accessibility and developer productivity.

---

## 🛠️ **Tech Stack**

- **Programming Language:** Python
- **Speech Recognition:** Whisper
- **Large Language Model (LLM):** LLaMA 3 (Groq/Local Deployment)
- **Shell Scripting:** Bash / Linux Terminal
- **Frameworks/Libraries:** 
  - `whisper` for transcription
  - `subprocess` for command execution
  - `transformers` for LLM integration

---

## 📂 **Project Structure**

GenOS/
├── main.py
├── whisper_transcribe.py
├── llm_command_generator.py
├── command_executor.py
├── requirements.txt
└── README.md

markdown
Copy
Edit

- **main.py** – Entry point integrating transcription, LLM processing, and execution.
- **whisper_transcribe.py** – Handles audio capture and transcription.
- **llm_command_generator.py** – Generates shell commands using LLaMA.
- **command_executor.py** – Executes validated commands on the OS.
- **requirements.txt** – Lists dependencies for easy setup.

---

## ⚙️ **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/shashidhar-N-18/GenOS.git
cd GenOS
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up Whisper and LLaMA

Ensure you have Whisper installed or configured via OpenAI API or local deployment.

Deploy LLaMA 3 using Groq or your preferred local environment (e.g., Ollama).

🧑‍💻 Usage
bash
Copy
Edit
python main.py
Speak your command after the voice prompt.

Whisper transcribes your input.

LLaMA processes and generates the corresponding shell command(s).

Commands are executed, and output is displayed.

📝 Example Commands
"Create a new folder called Projects and open it."
➔ Generates: mkdir Projects && cd Projects

"List all files modified today in this directory."
➔ Generates: find . -type f -daystart -mtime -1

💡 Future Enhancements
GUI overlay for command suggestions and confirmations.

Multi-user voice profiles for personalized command history.

Integration with VS Code or IDEs for direct code execution.

Remote device orchestration via SSH.

