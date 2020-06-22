from flask import Flask, request
import json
import relation

app = Flask(__name__)

app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))


@app.route("/relation_extract", methods=['GET', 'POST'])
def extract_sentence_from_req():
    sentence = request.form.get('sentence')
    rst = relation.extract_all_relations(sentence)
    return json.dumps(rst)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=12315)
