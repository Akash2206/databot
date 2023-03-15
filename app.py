from flask import Flask, request, render_template
import pandas as pd
import main

app = Flask(__name__)
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
    
@app.route('/process_csv', methods=['POST'])
def process_csv():
    if request.method == 'POST':
        csvfile = request.files['csvfile1']
        target = request.form['param1']
        test_size = float(request.form['param2'])
        df = pd.read_csv(csvfile)
        output = main.run(df, target, test_size)
    return render_template('index.html', output=output.to_html())

if __name__ == '__main__':
    app.run(debug=True)
