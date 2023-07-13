import pandas as pd
import openai

# Load the Excel sheet containing the acronyms and definitions
df = pd.read_excel('abbreviations.xlsx')

# Create a dictionary mapping acronyms to definitions
acronym_definitions = dict(zip(df['Acronym'], df['Definition']))

# OpenAI API credentials
openai.api_key = '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'

# Define a function to get the definition of an acronym
def get_definition(acronym):
    if acronym in acronym_definitions:
        return acronym_definitions[acronym]
    else:
        return "Sorry, I don't have a definition for that acronym."


# Define a function to generate a response using OpenAI
def generate_response(user_input):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=f"What does {user_input} mean?",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()


# Chatbot loop
while True:
    user_input = input("User: ")

    if user_input.lower() == 'exit':
        break

    if user_input.isupper():
        acronym = user_input.upper()
        definition = get_definition(acronym)
        if definition:
            print(f"Chatbot: {definition}")
        else:
            print(f"Chatbot: Sorry, I don't have a definition for the acronym {acronym}.")
    else:

        response = generate_response(user_input)
        print(f"Chatbot: {response}")
