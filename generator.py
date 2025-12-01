import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import markdown
from fpdf import FPDF
import argparse
import magic  # For file type detection
import chardet  # For encoding detection
import g4f
import yaml
# Load environment variables
load_dotenv()
def normalize_path(path: str) -> str:
    """Normalize path to use forward slashes"""
    return path.replace('\\', '/')
class LanguageDetector:
    """Detects programming language of source files."""
    
    LANGUAGE_SIGNATURES = {
        'python': [
            (r'^#!.*python', re.IGNORECASE),
            (r'\bimport\s+\w+|from\s+\w+\s+import\b', re.IGNORECASE),
            (r'\bdef\s+\w+\s*\(|class\s+\w+', re.IGNORECASE),
            (r'\bprint\s*\(|\breturn\s+', re.IGNORECASE)
        ],
        'javascript': [
            (r'^#!.*node', re.IGNORECASE),
            (r'\bfunction\s+\w+\s*\(|const\s+\w+\s*=|let\s+\w+\s*=', re.IGNORECASE),
            (r'\bconsole\.log\s*\(|\bexport\s+', re.IGNORECASE)
        ],
        'java': [
            (r'\bpublic\s+class\s+\w+', re.IGNORECASE),
            (r'\bpublic\s+\w+\s+\w+\s*\(', re.IGNORECASE),
            (r'\bSystem\.out\.println\s*\(', re.IGNORECASE),
            (r'\bimport\s+[\w\.]+\s*;', re.IGNORECASE)
        ],
        'c': [
            (r'^#include\s+<[\w\.]+>', re.IGNORECASE),
            (r'\bint\s+main\s*\(', re.IGNORECASE),
            (r'\bprintf\s*\(|\breturn\s+', re.IGNORECASE)
        ],
        'cpp': [
            (r'^#include\s+<[\w\.]+>', re.IGNORECASE),
            (r'\bstd::\w+', re.IGNORECASE),
            (r'\bclass\s+\w+', re.IGNORECASE),
            (r'\bcout\s*<<|\bcin\s*>>', re.IGNORECASE)
        ],
        'go': [
            (r'^package\s+\w+', re.IGNORECASE),
            (r'\bfunc\s+\w+\s*\(', re.IGNORECASE),
            (r'\bimport\s*\([^)]*\)', re.IGNORECASE),
            (r'\bfmt\.Println\s*\(', re.IGNORECASE)
        ]
    }
    
    @staticmethod
    def detect_language(file_path: str) -> Optional[str]:
        """Detect the programming language of a file using multiple methods."""
        # First try by file extension
        ext_lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'cpp',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.ts': 'typescript'
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ext_lang_map:
            return ext_lang_map[ext]
        
        # If extension is ambiguous or missing, try content analysis
        try:
            # Detect file encoding first
            with open(file_path, 'rb') as f:
                raw_data = f.read(4096)  # Read first 4KB for analysis
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            
            # Read file content with detected encoding
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read(2048)  # Read first 2KB for analysis
                
                # Check for shebang
                first_line = content.split('\n')[0] if content else ''
                if first_line.startswith('#!'):
                    if 'python' in first_line.lower():
                        return 'python'
                    elif 'node' in first_line.lower():
                        return 'javascript'
                    elif 'bash' in first_line.lower():
                        return 'bash'
                
                # Check language signatures
                for lang, patterns in LanguageDetector.LANGUAGE_SIGNATURES.items():
                    for pattern, flags in patterns:
                        if re.search(pattern, content, flags):
                            return lang
                
                # Try libmagic as fallback
                mime = magic.from_file(file_path, mime=True)
                if mime:
                    if 'x-python' in mime:
                        return 'python'
                    elif 'javascript' in mime.lower():
                        return 'javascript'
                    elif 'java' in mime.lower():
                        return 'java'
                    elif 'c++' in mime.lower() or 'c source' in mime.lower():
                        return 'cpp'
        
        except Exception as e:
            print(f"Error detecting language for {file_path}: {str(e)}")
        
        return None


class CodeParser:
    """Parses source code files to extract structural information."""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
    
    def parse_file(self, file_path: str) -> Optional[Dict]:
        """Parse a source code file based on its detected language."""
        language = self.language_detector.detect_language(file_path)
        
        if not language:
            print(f"Could not detect language for: {file_path}")
            return None
        
        if language == 'python':
            return self._parse_python_file(file_path)
        elif language == 'javascript':
            return self._parse_javascript_file(file_path)
        elif language == 'java':
            return self._parse_java_file(file_path)
        # Add parsers for other languages here
        else:
            print(f"No parser available for {language} files")
            return None
    
    def _parse_python_file(self, file_path: str) -> Dict:
        """Parse a Python file and extract functions, classes, and docstrings."""
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        
        tree = ast.parse(code)
        
        elements = {
            'file_name': os.path.basename(file_path),
            'language': 'python',
            'classes': [],
            'functions': [],
            'imports': [],
            'module_doc': ast.get_docstring(tree)
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'methods': []
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'docstring': ast.get_docstring(item),
                            'args': [arg.arg for arg in item.args.args]
                        }
                        class_info['methods'].append(method_info)
                
                elements['classes'].append(class_info)
            
            elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                function_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'args': [arg.arg for arg in node.args.args]
                }
                elements['functions'].append(function_info)
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    elements['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for alias in node.names:
                    elements['imports'].append(f"{module}.{alias.name}")
        
        return elements
    
    def _parse_javascript_file(self, file_path: str) -> Dict:
        """Parse a JavaScript file and extract functions, classes, and docstrings."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # This is a simplified JavaScript parser - consider using a proper parser like esprima for production
        elements = {
            'file_name': os.path.basename(file_path),
            'language': 'javascript',
            'classes': [],
            'functions': [],
            'imports': [],
            'module_doc': ''
        }
        
        # Extract JSDoc comments (simplified)
        jsdoc_pattern = r'/\*\*([^*]|(\*(?!/))+\*/'
        jsdocs = {}
        for match in re.finditer(jsdoc_pattern, content):
            comment = match.group()
            # Find the next function/class after this comment
            next_code = content[match.end():match.end()+100]
            if 'function ' in next_code or 'class ' in next_code:
                name_match = re.search(r'(function|class)\s+(\w+)', next_code)
                if name_match:
                    jsdocs[name_match.group(2)] = comment
        
        # Find classes
        class_pattern = r'class\s+(\w+)\s*{([^}]*)}'
        for match in re.finditer(class_pattern, content, re.DOTALL):
            class_name = match.group(1)
            class_body = match.group(2)
            
            class_info = {
                'name': class_name,
                'docstring': jsdocs.get(class_name, ''),
                'methods': []
            }
            
            # Find methods in class
            method_pattern = r'(\w+)\s*\(([^)]*)\)\s*{'
            for method_match in re.finditer(method_pattern, class_body):
                method_name = method_match.group(1)
                method_args = [arg.strip() for arg in method_match.group(2).split(',') if arg.strip()]
                
                method_info = {
                    'name': method_name,
                    'docstring': jsdocs.get(method_name, ''),
                    'args': method_args
                }
                class_info['methods'].append(method_info)
            
            elements['classes'].append(class_info)
        
        # Find functions
        function_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*{'
        for match in re.finditer(function_pattern, content):
            func_name = match.group(1)
            func_args = [arg.strip() for arg in match.group(2).split(',') if arg.strip()]
            
            function_info = {
                'name': func_name,
                'docstring': jsdocs.get(func_name, ''),
                'args': func_args
            }
            elements['functions'].append(function_info)
        
        # Find arrow functions assigned to variables (simplified)
        arrow_pattern = r'const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>'
        for match in re.finditer(arrow_pattern, content):
            func_name = match.group(1)
            func_args = [arg.strip() for arg in match.group(2).split(',') if arg.strip()]
            
            function_info = {
                'name': func_name,
                'docstring': jsdocs.get(func_name, ''),
                'args': func_args
            }
            elements['functions'].append(function_info)
        
        # Find imports
        import_pattern = r'import\s+(?:{[^}]+}\s+from\s+)?[\'"]([^\'"]+)[\'"]'
        elements['imports'] = list(set(re.findall(import_pattern, content)))
        
        return elements
    
    def _parse_cpp_file(self, file_path: str) -> Dict:
        """Basic C++ file parser."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        elements = {
            'file_name': os.path.basename(file_path),
            'language': 'cpp',
            'classes': [],
            'functions': [],
            'imports': [],
            'module_doc': ''
        }
        
        # Find #include statements
        elements['imports'] = re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)
        
        # Find class definitions
        class_pattern = r'class\s+(\w+)\s*{([^}]*)}'
        for match in re.finditer(class_pattern, content, re.DOTALL):
            class_info = {
                'name': match.group(1),
                'docstring': '',
                'methods': []
            }
            elements['classes'].append(class_info)
        
        # Find function definitions
        function_pattern = r'\w+\s+\w+\s*\(([^)]*)\)\s*{'
        for match in re.finditer(function_pattern, content):
            func_name = match.group(0).split('(')[0].split()[-1]
            function_info = {
                'name': func_name,
                'docstring': '',
                'args': [arg.strip() for arg in match.group(1).split(',') if arg.strip()]
            }
            elements['functions'].append(function_info)
        
        return elements

    def _parse_java_file(self, file_path: str) -> Dict:
        """Parse a Java file and extract methods, classes, and docstrings."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        elements = {
            'file_name': os.path.basename(file_path),
            'language': 'java',
            'classes': [],
            'functions': [],
            'imports': [],
            'module_doc': ''
        }
        
        # Extract Javadoc comments
        javadoc_pattern = r'/\*\*([^*]|(\*(?!/))+\*/'
        javadocs = {}
        for match in re.finditer(javadoc_pattern, content):
            comment = match.group()
            # Find the next class/method after this comment
            next_code = content[match.end():match.end()+100]
            if 'class ' in next_code or re.search(r'(public|private|protected)\s+\w+\s+\w+\s*\(', next_code):
                name_match = re.search(r'class\s+(\w+)', next_code) or \
                             re.search(r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(', next_code)
                if name_match:
                    javadocs[name_match.group(1)] = comment
        
        # Find classes
        class_pattern = r'class\s+(\w+)\s*{([^}]*)}'
        for match in re.finditer(class_pattern, content, re.DOTALL):
            class_name = match.group(1)
            class_body = match.group(2)
            
            class_info = {
                'name': class_name,
                'docstring': javadocs.get(class_name, ''),
                'methods': []
            }
            
            # Find methods in class
            method_pattern = r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(([^)]*)\)'
            for method_match in re.finditer(method_pattern, class_body):
                method_name = method_match.group(1)
                method_args = [arg.strip() for arg in method_match.group(2).split(',') if arg.strip()]
                
                method_info = {
                    'name': method_name,
                    'docstring': javadocs.get(method_name, ''),
                    'args': method_args
                }
                class_info['methods'].append(method_info)
            
            elements['classes'].append(class_info)
        
        # Find imports
        import_pattern = r'import\s+([\w\.]+)\s*;'
        elements['imports'] = list(set(re.findall(import_pattern, content)))
        
        return elements


class DocumentationGenerator:
    """Generates documentation using GPT-4 and formats the output."""
    
    def __init__(self: str):
        self.code_parser = CodeParser()
        self.template = """
        # {element_name}
        
        {description}
        
        {usage_example}
        
        ## Parameters
        {parameters}
        
        ## Returns
        {returns}
        """
    
    def generate_documentation(self, code_info: Dict) -> str:
        """Generate documentation for a code element using GPT-4 with fallback to GPT-3.5-turbo."""
        prompt = f"""
        Generate comprehensive technical documentation for the following {code_info.get('language', '')} code element:
        
        Name: {code_info.get('name', '')}
        Type: {code_info.get('type', '')}
        File: {code_info.get('file_name', '')}
        
        Docstring: {code_info.get('docstring', 'No docstring available')}
        
        Parameters: {', '.join(code_info.get('args', []))}
        
        Please provide:
        1. A clear description of what this element does
        2. Explanation of each parameter
        3. Return value description (if applicable)
        4. A usage example
        5. Any important notes
        
        Format your response in Markdown and include language-specific conventions.
        """
        
        last_error = None
        
        
        try:
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4,
                messages=[
                    {"role": "system", "content": "You are a technical documentation writer."},
                    {"role": "user", "content": prompt}
                ],
                provider=g4f.Provider.Copilot,
                temperature=0.3,
                max_tokens=1500
            )
            print("chatgpt response:")
            print(response)
            return response
                
        except Exception as e:
            last_error = e 
        # If model failed
        print(f"Error: : {last_error}")
        
        # Return a basic documentation template as fallback
        return self._generate_basic_docs_fallback(code_info)

    def _generate_basic_docs_fallback(self, code_info: Dict) -> str:
        """Generate basic documentation when API calls fail."""
        params = "\n".join(f"- `{param}`" for param in code_info.get('args', []))
        
        return f"""
        # {code_info.get('name', 'Unknown')}

        **File:** `{code_info.get('file_name', '')}`

        ## Description
        {code_info.get('docstring', 'No documentation available')}

        ## Parameters
        {params if params else "None"}

        ## Returns
        Not documented
        """
    def format_documentation(self, raw_docs: str, code_info: Dict) -> str:
        """Format the generated documentation into a consistent structure without extra indentation."""
        # Extract sections from the raw documentation
        sections = {
            'description': '',
            'usage_example': '',
            'parameters': '',
            'returns': ''
        }
        
        current_section = None
        for line in raw_docs.split('\n'):
            if line.startswith('## '):
                section_name = line[3:].strip().lower()
                if section_name in sections:
                    current_section = section_name
                else:
                    current_section = None
            elif current_section:
                sections[current_section] += line + '\n'
        
        # Clean up each section by stripping whitespace
        for key in sections:
            sections[key] = sections[key].strip()
        
        # Format parameters if they exist
        params_section = ""
        if code_info.get('args'):
            params_section = "\n".join(
                f"- `{param}`: {self._extract_param_description(param, sections['parameters'])}"
                for param in code_info['args']
            )
        
        # Create the documentation without extra indentation
        docs = f"""# {code_info['name']}

    **File:** `{code_info['file_name']}`  
    **Language:** `{code_info.get('language', 'unknown')}`

    {sections['description']}"""

        # Add usage example if it exists
        if sections['usage_example']:
            docs += f"""

    ## Usage Example
    {sections['usage_example']}"""
        
        # Add parameters section
        docs += f"""

    ## Parameters
    {params_section if params_section else 'None'}"""
        
        # Add returns section
        docs += f"""

    ## Returns
    {sections['returns'] if sections['returns'] else 'Not documented'}"""
        
        return docs
    
    def _extract_param_description(self, param_name: str, parameters_text: str) -> str:
        """Extract description for a specific parameter."""
        pattern = re.compile(rf"`?{param_name}`?[\s:]+(.+?)(?=\n\s*-|\n\s*`|\n\s*##|$)", re.IGNORECASE)
        match = pattern.search(parameters_text)
        return match.group(1).strip() if match else "No description available"
    
    def generate_file_documentation(self, file_path: str, output_format: str = "markdown") -> str:
        """Generate documentation for an entire file."""
        code_info = self.code_parser.parse_file(file_path)
        if not code_info:
            return ""
        
        # Generate module-level documentation
        full_docs = f"# File: {code_info['file_name']}\n\n"
        full_docs += f"**Language:** `{code_info.get('language', 'unknown')}`\n\n"
        
        if code_info.get('module_doc'):
            full_docs += f"## Overview\n{code_info['module_doc']}\n\n"
        
        if code_info.get('imports'):
            full_docs += "## Imports\n" + "\n".join(f"- `{imp}`" for imp in code_info['imports']) + "\n\n"
        
        # Generate documentation for classes
        for class_info in code_info['classes']:
            class_info['type'] = 'class'
            class_info['file_name'] = code_info['file_name']
            class_info['language'] = code_info.get('language', 'unknown')
            raw_docs = self.generate_documentation(class_info)
            formatted_docs = self.format_documentation(raw_docs, class_info)
            full_docs += formatted_docs + "\n\n"
            
            # Generate documentation for methods
            for method_info in class_info['methods']:
                method_info['type'] = 'method'
                method_info['file_name'] = code_info['file_name']
                method_info['language'] = code_info.get('language', 'unknown')
                method_info['name'] = f"{class_info['name']}.{method_info['name']}"
                raw_docs = self.generate_documentation(method_info)
                formatted_docs = self.format_documentation(raw_docs, method_info)
                full_docs += formatted_docs + "\n\n"
        
        # Generate documentation for standalone functions
        for func_info in code_info['functions']:
            func_info['type'] = 'function'
            func_info['file_name'] = code_info['file_name']
            func_info['language'] = code_info.get('language', 'unknown')
            raw_docs = self.generate_documentation(func_info)
            formatted_docs = self.format_documentation(raw_docs, func_info)
            full_docs += formatted_docs + "\n\n"
        
        if output_format == "pdf":
            return self._generate_pdf(full_docs, code_info['file_name'])
        return full_docs
    
    def generate_mkdocs_site(self, input_path: str):
        """Generate complete MkDocs site from source code."""
        # Ensure proper directory structure exists
        docs_content_dir = os.path.join(self.output_dir, "docs")
        os.makedirs(docs_content_dir, exist_ok=True)
        
        mkdocs = MkDocsGenerator(self.output_dir)
        mkdocs.initialize_mkdocs()
        
        if os.path.isfile(input_path):
            docs = self.generate_file_documentation(input_path)
            output_file = os.path.join(
                docs_content_dir,
                f"{os.path.splitext(os.path.basename(input_path))[0]}.md"
            )
            print((f"{os.path.splitext(os.path.basename(input_path))[0]}.md"))
            with open(output_file, 'w') as f:
                f.write(docs)
            mkdocs.add_to_nav(os.path.basename(input_path),  (f"{os.path.splitext(os.path.basename(input_path))[0]}.md"))
        else:
            for root, _, files in os.walk(input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        docs = self.generate_file_documentation(file_path)
                        rel_path = os.path.relpath(root, input_path)
                        output_dir = os.path.join(docs_content_dir, rel_path)
                        os.makedirs(output_dir, exist_ok=True)
                    
                        output_file = os.path.join(
                            output_dir,
                            f"{os.path.splitext(file)[0]}.md"
                        )
                    
                        with open(output_file, 'w') as f:
                            f.write(docs)
                    
                        nav_name = os.path.join(os.path.splitext(file)[0])
                        
                        mkdocs.add_to_nav(nav_name, os.path.join(f"{os.path.splitext(file)[0]}.md"))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        mkdocs.build_site()
    def _generate_pdf(self, markdown_text: str, filename: str) -> str:
    
        try:
            # Convert markdown to HTML
            html = markdown.markdown(markdown_text)
            
            # Create PDF with better configuration
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            
            # Better HTML handling
            for line in html.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Handle headings
                if line.startswith('<h1>'):
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, line[4:-5].strip(), ln=True)
                    pdf.set_font("Arial", size=12)
                elif line.startswith('<h2>'):
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, line[4:-5].strip(), ln=True)
                    pdf.set_font("Arial", size=12)
                # Handle lists
                elif line.startswith('<li>'):
                    pdf.cell(10)  # Indent
                    pdf.cell(0, 10, "• " + line[4:-5].strip(), ln=True)
                # Handle paragraphs and other content
                else:
                    # Remove any HTML tags that might remain
                    text = re.sub('<[^<]+?>', '', line)
                    pdf.multi_cell(0, 10, text)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate output path
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_path = os.path.join(output_dir if output_dir else "", f"{base_name}_docs.pdf")
            
            pdf.output(output_path)
            return output_path
        
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return ""

class MkDocsGenerator:
    """Handles MkDocs integration and site generation."""
    
    def __init__(self, docs_dir: str = "docs"):
         # Normalize the path to avoid duplicates
        self.docs_dir = os.path.normpath(docs_dir)
       
        self.mkdocs_config = os.path.join(self.docs_dir, "mkdocs.yml")
        
        # Ensure the directory exists
        
        os.makedirs(self.docs_dir, exist_ok=True)
        self.server_process = None
        
    def initialize_mkdocs(self):
        """Initialize MkDocs directory structure if it doesn't exist."""
        # Ensure proper directory structure exists
        docs_content_dir = os.path.join(self.docs_dir, "docs")
        os.makedirs(docs_content_dir, exist_ok=True)
        
        # Create proper YAML content with correct indentation
        base_config = """site_name: Code Documentation
theme: readthedocs
nav:
- Home: index.md
"""
        
        # Write the config file with proper formatting
        with open(self.mkdocs_config, 'w', encoding='utf-8') as f:
            f.write(base_config)
        
        # Create default index.md if it doesn't exist
        index_path = os.path.join(docs_content_dir, "index.md")
        if not os.path.exists(index_path):
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write("# Code Documentation\n\nGenerated from source code.")
    
    def add_to_nav(self, section_name: str, doc_path: str):
        """Add a documentation file to MkDocs navigation."""
        try:
            
            # Load existing config
            with open(self.mkdocs_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Initialize nav if it doesn't exist
            if 'nav' not in config:
                config['nav'] = []
            
            # Add new entry if not already present
            new_entry = {section_name: doc_path}
            if new_entry not in config['nav']:
                config['nav'].append(new_entry)
            
            # Write back with proper YAML formatting
            with open(self.mkdocs_config, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
        except Exception as e:
            print(f"Error updating navigation: {e}")
        
    def build_site(self):
        """Build the MkDocs site."""
        import subprocess
        try:
            subprocess.run(["mkdocs", "build"], check=True, cwd=self.docs_dir)
            print("MkDocs site built successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error building MkDocs site: {e}")
        except FileNotFoundError:
            print("MkDocs is not installed. Please install with: pip install mkdocs")
    def serve_site(self, host: str = "127.0.0.1", port: int = 8000):
        """Serve the MkDocs site locally with auto-reload."""
        import subprocess
        import threading
        import time
        import webbrowser
        
        def run_server():
            try:
                subprocess.run(
                    ["mkdocs", "serve", "--dev-addr", f"{host}:{port}"],
                    cwd=self.docs_dir,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"MkDocs server error: {e}")
            except KeyboardInterrupt:
                print("\nStopping MkDocs server...")
        
        # Open browser after short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open_new_tab(f"http://{host}:{port}")
        
        print(f"\nStarting MkDocs server at http://{host}:{port}")
        print("Press Ctrl+C to stop the server\n")
        
        # Start browser thread
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Start server in main thread
        try:
            run_server()
        except KeyboardInterrupt:
            print("Server stopped")

def process_directory(input_path: str, output_path: str , api_key: str,format: str = "markdown"):
    """Process all source files in a directory and generate documentation."""
    doc_generator = DocumentationGenerator()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for root, _, files in os.walk(input_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing {file_path}...")
            
            try:
                docs = doc_generator.generate_file_documentation(file_path, format)
                
                if format == "pdf":
                    # PDFs are already saved by the generator
                    continue
                
                output_file = os.path.join(
                    output_path,
                    f"{os.path.splitext(file)[0]}_docs.md"
                )
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(docs)
            
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")


def main():
    g4f.debug.logging = True # enable logging
    parser = argparse.ArgumentParser(description="Generate documentation from source code using GPT-4")
    parser.add_argument('--input', required=True, help="Input file or directory path")
    parser.add_argument('--output', default="docs", help="Output directory path")
    parser.add_argument('--format', choices=['markdown',  'mkdocs'], default='markdown',
                      help="Output format (markdown, pdf, or mkdocs)")
    parser.add_argument('--serve', action='store_true', 
                      help="Serve the documentation after generation (mkdocs format only)")
    parser.add_argument('--host', default="127.0.0.1", 
                      help="Host address to serve documentation (default: 127.0.0.1)")
    parser.add_argument('--port', type=int, default=8000,
                      help="Port to serve documentation (default: 8000)")
    
    args = parser.parse_args()
    

    
    doc_generator = DocumentationGenerator()
    doc_generator.output_dir = args.output
    
    if args.format == "mkdocs":
        doc_generator.generate_mkdocs_site(args.input)
        if args.serve:
            mkdocs = MkDocsGenerator(args.output)
            mkdocs.serve_site(host=args.host, port=args.port)
    elif os.path.isfile(args.input):
        # Process single file
        doc_generator = DocumentationGenerator()
        docs = doc_generator.generate_file_documentation(args.input, args.format)
        
        if args.format == "pdf":
            print(f"PDF documentation generated at {docs}")
        else:
            output_file = os.path.join(
                args.output,
                f"{os.path.splitext(os.path.basename(args.input))[0]}_docs.md"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(docs)
            print(f"Markdown documentation generated at {output_file}")
    else:
        # Process directory
        process_directory(args.input, args.output, args.format, api_key)
        print(f"Documentation generated in {args.output}")


if __name__ == "__main__":
    main()