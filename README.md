# Generating-technical-documentation-from-code-using-GPT-4
Automatically generate technical documentation from source code using GPT-4. This project converts functions, classes, and modules into clean Markdown or PDF documentation using the OpenAI API. It integrates with tools like Sphinx and MkDocs without requiring a GPU, and is built with Python for easy automation and extensibility.


# How to run
First, run the provided commands in the terminal:

pip install -r requirements.txt

Then run:

generator.py [-h] --input INPUT [--output OUTPUT] [--format {markdown,mkdocs}] [--serve]
