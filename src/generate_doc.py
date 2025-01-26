TICKER_SYMBOL = "NVDA"
REC = -1

ticker_string = f"#let ticker_symbol = \"{TICKER_SYMBOL}\""
rec_string = f"#let recommendation = {REC}"

full_substitution = ticker_string + "\n" + rec_string
with open('src/doc_template.typ', 'r') as file:
    doc_template = file.read()

full_doc = doc_template.replace("//substitute", full_substitution)

with open('output.typ', 'w') as file:
    file.write(full_doc)
