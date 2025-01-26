TICKER_SYMBOL = "NVDA"
REC = -1

ticker_string = f'#let ticker_symbol = "{TICKER_SYMBOL}"'
rec_string = f"#let recommendation = {REC}"

full_substitution = ticker_string + "\n" + rec_string
with open("src/doc_template.typ", "rb") as file:
    doc_template = file.read()

template_str = str(doc_template, encoding="utf-8")

full_doc = template_str.replace("//substitute", full_substitution)

with open("output.typ", "wb") as file:
    file.write(full_doc.encode("utf-8"))
