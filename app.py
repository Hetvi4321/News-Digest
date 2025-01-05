from flask import Flask, request, jsonify, render_template
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

def run_notebook(notebook_path, parameters):
    """Executes a Jupyter Notebook and retrieves output."""
    try:
        # Open and read the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Inject parameters into the first cell of the notebook
        param_cell = nbformat.v4.new_code_cell(source=parameters)
        nb.cells.insert(0, param_cell)

        # Set up the notebook execution
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': './'}})

        return nb, None  # Return notebook and no error

    except Exception as e:
        logging.error(f"Error executing notebook: {str(e)}")
        return None, str(e)  # Return None and the error message

@app.route('/')
def index():
    """Renders the user interface."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handles user input, runs notebooks, and returns summarized news."""
    category = request.json.get('category')

    if not category:
        return jsonify({'error': 'Category is required'}), 400

    # Prepare parameters for findingnews.ipynb
    findingnews_params = f"category = '{category}'\n"

    # Run findingnews.ipynb
    findingnews_nb, error = run_notebook('findingnews.ipynb', findingnews_params)
    if error:
        return jsonify({'error': 'Error running findingnews.ipynb: ' + error}), 500

    # Access findingnews output based on keys (reliable & flexible)
    findingnews_output = {}
    for cell in findingnews_nb.cells:
        if cell.cell_type == 'code' and 'outputs' in cell:
            for output in cell.outputs:
                if output.output_type == 'stream' and output.name == 'stdout':
                    try:
                        findingnews_output = json.loads(output.text.strip())
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON output: {str(e)}")
                    break  # Stop after finding the first relevant output

    # Check for errors or missing data
    if 'error' in findingnews_output:
        return jsonify({'error': findingnews_output['error']}), 500

    if not all(key in findingnews_output for key in ('headline', 'combined_text', 'sources')):
        return jsonify({'error': 'Missing data in findingnews output'}), 500

    # Prepare parameters for summarize.ipynb
    summarize_params = f"text_to_summarize = '''{findingnews_output['combined_text']}'''\n"
    logging.debug(f"summarize_params: {summarize_params}")  # Debug log to check parameters

    # Run summarize.ipynb
    summarize_nb, error = run_notebook('summarize.ipynb', summarize_params)
    if error:
        return jsonify({'error': 'Error running summarize.ipynb: ' + error}), 500

    # Access summarize output (similar approach)
    summarize_output = {}
    for cell in summarize_nb.cells:
        if cell.cell_type == 'code' and 'outputs' in cell:
            for output in cell.outputs:
                if output.output_type == 'stream' and output.name == 'stdout':
                    try:
                        summarize_output = json.loads(output.text.strip())
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON output: {str(e)}")
                    break

    # Check for errors or missing data
    if 'error' in summarize_output:
        return jsonify({'error': summarize_output['error']}), 500

    return jsonify({
        'headline': findingnews_output['headline'],
        'sources': findingnews_output['sources'],
        'summary': summarize_output['summary']
    })

if __name__ == '__main__':
    app.run(debug=True)
