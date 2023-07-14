from flask import Flask, render_template, request, url_for, flash, redirect
from backend import run_chatbot

app = Flask(__name__)
#Not so secret, secret key
app.config['SECRET_KEY'] = '1be50495bed1fc45e20ec51d30d51be18ce92026ed4483d3'

messages = []
@app.route("/", methods=('GET', 'POST'))
def chat_bot():
    if request.method == 'GET':
        return render_template("app_home.html", messages = messages)
    elif request.method == 'POST':
        question = request.form['question']
        if question == "":
            flash('Please type something bozo')
        else:
            # GET ANSWER from CHATGPT
            answer = run_chatbot(question)
            print(answer)
            if len(messages) > 0:
                del messages[-1]['last']
            messages.append({'question': question, 'answer': answer, 'last': True})
            return redirect(request.url+"#last_message")
    return render_template("app_home.html", messages = messages)



if __name__ == "__main__":
    app.run(debug=True)

