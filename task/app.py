from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role
from task._constants import API_KEY


# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """
You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question and politly ask user to provide context.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""

# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)



# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml

def main():
  db_conf = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
  }

  conversation = Conversation()
  conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))
  print(f"{'='*30} Text Processing {'='*30}")
  embeddings_client = DialEmbeddingsClient(deployment_name='text-embedding-3-small-1')
  text_processor = TextProcessor(embeddings_client, db_conf, dimensions=1536)
  # cwd = __file__.rsplit('/', 1)[0]
  # text_processor.process_text_file(
  #   file_name=f'{cwd}/embeddings/microwave_manual.txt',
  #   chunk_size=300,
  #   overlap=25,
  #   truncate_table=True
  # )
  print(f"{'='*80}")

  while True:
    user_input = input("User: ")
    if user_input.lower() in ['/exit', '/quit']:
      print("Exiting chat. Goodbye!")
      break

    if not user_input.strip():
      print("Please enter a valid question.")
      continue

    print(f"{'='*30} Searching {'='*30}")
    search_result = text_processor.search(user_request=user_input, search_mode=SearchMode.COSINE_DISTANCE)

    argumentations = [text for text, _score in search_result]
    user_prompt = USER_PROMPT.format(context="\n\n".join(argumentations), query=user_input)

    print(f"{'='*80}\n{'='*30} Argumenation {'='*30}\n{user_prompt}\n{'='*80}\n{'='*30} Generation {'='*30}")
    conversation.add_message(Message(Role.USER, user_prompt))
    chat_client = DialChatCompletionClient(deployment_name='gpt-4o',api_key=API_KEY)
    message = chat_client.get_completion(conversation.messages, print_request=True)
    conversation.add_message(message)
    print(f"{'='*80}\n\nAssistant: {message.content}")


if __name__ == "__main__":
  main()
