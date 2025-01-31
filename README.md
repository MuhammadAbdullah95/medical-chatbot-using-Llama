# End-to-end-Medical-Chatbot-using-Llama2

# How to run?

### STEPS:

Clone the repository

```bash
Project repo: https://github.com/MuhammadAbdullah95/medical-chatbot-using-Llama.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Use LLMs Like Llama, Deepseek, Google Gemini, OpenAi etc.

## Set Credentials as follows in `.env` file

```ini

GEMINI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxx"

```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up localhost:
```

### Techstack Used:

- Python
- LangChain
- Flask
- Llama 3.2
- Pinecone
