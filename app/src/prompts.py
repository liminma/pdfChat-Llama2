llama2_prompt_ending_words = 'Helpful answer:'

llama2_template = """Use the context below to answer questions. if you're not sure of an answer, you can say "I don't know".

Context: {context}

Question: {question}

Only return the helpful answer below and nothing else.
%s""" % llama2_prompt_ending_words